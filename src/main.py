"""
Phyiscs inofrmed unsupervised approach for automated de-noiseing of Partial Discharge signals.

Author: Lewis Green
Date: 2024-06-15
MOINTRA 
"""
import numpy as np
import pandas as pd
import sqlite3
import pathlib 
import json 
import time 

import torch
import torch.nn as nn
import torch.nn.functional as F 

import matplotlib.pyplot as plt

from utils import plot_clusters
from data_preperation import getSensor, getEventCount, getNComponents, normaliseDataset, inverseTransform
from autoencoder import PhysicsInformedAutoencoder, train_pi_ae
from feature_extraction import get_adaptive_thresholds, extract_pd_features, normalise_features, get_feature_thresholds
from clustering import hdbscan, isolationForest, min_cluster_calc
from pd_selector import writeResults, assignWeights, aggregate_cluster_features, classify_clusters, computeScores, map_labels_to_events

seed = 42 
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def load_config(configPath: pathlib.Path):

    with configPath.open() as f: 
        cfg = json.load(f)
   
    # create dictionary of cfg contents. Access these at a later date
    configDict = {
        "databaseFile": cfg.get("databaseFile"),
        "channelNumber": cfg.get('channelNumber'),
        "startTime": cfg.get('startTime'),
        "endTime": cfg.get('endTime')
     }

    # Add explicit checks for critical keys
    if configDict["databaseFile"] is None:
        raise ValueError(f"Configuration file '{configPath}' is missing 'databaseFile' key.")
    if configDict["channelNumber"] is None:
        raise ValueError(f"Configuration file '{configPath}' is missing 'channelNumber' key.")
    if configDict["startTime"] is None:
        raise ValueError(f"Configuration file '{configPath}' is missing 'startTime' key.")
    if configDict["endTime"] is None:
        raise ValueError(f"Configuration file '{configPath}' is missing 'endTime' key.")

    return configDict

def main(configPath: pathlib.Path):
    start_time = time.time()
    # =======================================================
    # Load Configuration and Dataset
    cfg = load_config(configPath)
    print("Configuration loaded successfully:")
    # normalise the raw data to get normal distribution 
    data_normalised, transformers = normaliseDataset(cfg)
    # calcualte n_components for pca for unmodified dataset
    # want to capture the variance of the actual data rather than the latent space 
    n_components = getNComponents(data_normalised.drop(columns=['id', 'acquisition_id']))
    # Reset index for consistent alignment throughout pipeline
    data_normalised = data_normalised.reset_index(drop=True)
    #data_normalised = data_normalised.sample(514500, random_state=seed).reset_index(drop=True)  # testing 
    sensor = getSensor(cfg)
    acqui_df = getEventCount(cfg)  
    # =======================================================
    # Feature Extraction and Normalisation
    # Domain specific feature extraction, kurtosis etc. 
    print("\n[1/6] Extracting domain features...")
    pd_features = extract_pd_features(data_normalised.drop(columns=['id', 'acquisition_id']))
    normalised_features, domain_transformers = normalise_features(pd_features)
    
    # =======================================================
    # Autoencoder Initialisation 
    # Select only the physical PD features (exclude id, acq_id columns)
    print(f'\n[2/6] Training Physics Informed Autoencoder...')
    ae_features = ['energy', 'modifiedFrequency', 'observedArea_mVns', 'observedFallTime_ns',
                    'observedPeakWidth_10pc_ns', 'observedPhaseDegrees',
                    'observedRiseTime_ns',  'observedTime_ms', 'peakValue']
    data_numeric = data_normalised[ae_features]
    
    signal_length = len(ae_features)
    print(f"  Using {signal_length} features for autoencoder")
    
    data_tensor = torch.tensor(data_numeric.values, dtype=torch.float32)
    if len(data_tensor.shape) == 2:
        data_tensor = data_tensor.unsqueeze(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'  Using device: {device}')
    data_tensor = data_tensor.to(device)   
    
    loader = torch.utils.data.DataLoader(
        data_tensor,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4) 
    
    autoencoder = PhysicsInformedAutoencoder(signal_length=signal_length).to(device)    
    optimiser = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=1e-3, steps_per_epoch=len(loader), epochs=20)
 
    train_pi_ae(autoencoder, loader, optimiser, scheduler, device=device, epochs=20)
    
    # Obtain latent space representation
    autoencoder.eval()
    with torch.no_grad():
        latent_space = autoencoder.encode(data_tensor).numpy()
        latent_df = pd.DataFrame(latent_space, columns=[f'latent_{i}' for i in range(latent_space.shape[1])])

    print(f'  Latent space shape: {latent_df.shape}')
    
    # =======================================================
    # Noise filtering and clustering 
    print(f'\n[3/6] Filtering outliers and clustering...')
    # TODO optimize min_cluster_size based on physics informed approach
    min_cluster = min_cluster_calc(acqui_df, cfg , frequency=50) 
    #min_cluster = 30
    min_samples = 15
    
    # Add original IDs for later mapping
    latent_df['id'] = data_normalised.index.values
    
    isolated_df = isolationForest(latent_df)

    clustered_df = hdbscan(
        isolated_df.drop(columns=['id']), 
        n_components=n_components, 
        min_cluster_size=min_cluster, 
        min_samples=min_samples, 
        metric='euclidean'
    )
    
    # Restore ID column
    clustered_df['id'] = isolated_df['id'].values

    # =======================================================
    # PD classification 
    print(f'\n[4/6] Classifying PD vs noise...')
  
    # Filter features to match clustered samples (after isolation forest)
    # Use the indices that survived isolation forest filtering
    filtered_indices = isolated_df.index.tolist()
    filtered_features = normalised_features.iloc[filtered_indices].reset_index(drop=True)
    clustered_df_reset = clustered_df.reset_index(drop=True)
    
    # Aggregate features at cluster level
    cluster_stats = aggregate_cluster_features(clustered_df_reset, filtered_features)
    
   # adaptive thresholding seems to have corrected the issue of no PD being detected
    thresholds = get_adaptive_thresholds(cluster_stats, percentile=50)
    weights = assignWeights(thresholds)

    # Compute scores for each cluster
    scores_df = computeScores(cluster_stats, thresholds, weights)
    
    # Classify clusters
    classification_df = classify_clusters(scores_df, score_threshold=0.3, min_votes=2)
    
    # Map classification back to events
    labeled_df = map_labels_to_events(clustered_df_reset, classification_df)
    
    # Restore original ID
    labeled_df['id'] = clustered_df['id'].values

    # =======================================================
    # Summary statistics
    print(f'\n[5/6] Pipeline Summary:')
    print(f'  Total events processed: {len(data_normalised)}')
    print(f'  Events after outlier removal: {len(clustered_df)}')
    print(f'  Clusters found: {clustered_df["cluster"].nunique()}')
    print(f'  PD events: {(labeled_df["is_pd"] == 1).sum()}')
    print(f'  Noise events: {(labeled_df["is_pd"] == 0).sum()}')
    
    # =======================================================
    # Map results back to original dataframe
    # Create mapping from original index to cluster/is_pd labels
    result_mapping = labeled_df[['id', 'cluster', 'is_pd']].copy()
    result_mapping = result_mapping.set_index('id')
    
    # Initialize with default values (outliers removed by IsolationForest)
    data_normalised['cluster'] = -1  # -1 = outlier/removed
    data_normalised['is_pd'] = 0     # 0 = not classified as PD
    
    # Map cluster labels back using index
    for idx in result_mapping.index:
        if idx in data_normalised.index:
            data_normalised.loc[idx, 'cluster'] = result_mapping.loc[idx, 'cluster']
            data_normalised.loc[idx, 'is_pd'] = result_mapping.loc[idx, 'is_pd']
    
    print(f'  Outliers (cluster=-1): {(data_normalised["cluster"] == -1).sum()}')
    data_normalised = inverseTransform(data_normalised, transformers)
    # =======================================================
    # Write Results to db 
    print(f'\n[6/6] Writing results to database...')
    print(data_normalised.head(1))
    plot_clusters(data_normalised)
    #writeResults(data_normalised, classification_df, cfg, configPath)
    
    print("\n" + "="*60)
    print("        AUTOMATED DE-NOISING COMPLETE")
    print("="*60)
    print(f'Total Time Taken: {(time.time() - start_time):.2f}s\n')

    return labeled_df    


if __name__ == "__main__":
    configPath = pathlib.Path("C:/Users/lewis.green/Desktop/holy grain/config.json")
    main(configPath)