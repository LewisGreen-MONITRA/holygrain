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


from data_preperation import getDataset, getNComponents, normaliseDataset
from autoencoder import PhysicsInformedAutoencoder, train_pi_ae
from feature_extraction import extract_pd_features, normalise_features,get_feature_thresholds, analyze_features 
from clustering import hdbscan, isolationForest, min_cluster_calc
from pd_selector import (aggregate_cluster_features, assignWeights, computeScores, 
                         classify_clusters, map_labels_to_events, writeResults)    

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
    data_normalised = data_normalised.sample(100000).reset_index(drop=True)
    # =======================================================
    # Feature Extraction and Normalisation
    # Domain specific feature extraction, kurtosis etc. 
    pd_features = extract_pd_features(data_normalised)
    normalised_features, domain_transformers = normalise_features(pd_features)

    # =======================================================
    # Autoencoder Initialisation 
    # Select only the physical PD features (exclude id, acq_id columns)
    print(f'Initialising Physics Informed Autoencoder...')
    features = ['energy', 'modifiedFrequency', 'observedArea_mVns', 'observedFallTime_ns',
                    'observedPeakWidth_10pc_ns', 'observedPhaseDegrees',
                    'observedRiseTime_ns',  'observedTime_ms', 'peakValue']
    data_numeric = data_normalised[features]
    
    # Detect actual signal length from data
    
    signal_length = len(features)
    print(f"Using {signal_length} features for autoencoder")
    
    data_tensor = torch.tensor(data_numeric.values, dtype=torch.float32)
    if len(data_tensor.shape) == 2:
        # Add channel dimension: (N, features) -> (N, 1, features)
        data_tensor = data_tensor.unsqueeze(1)
    
    loader = torch.utils.data.DataLoader(
        data_tensor,
        batch_size=32,
        shuffle=True,
        drop_last=True) 
    autoencoder = PhysicsInformedAutoencoder(signal_length=signal_length)    
    optimiser = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    # Training      
    train_pi_ae(autoencoder, loader, optimiser, device=torch.device("cpu"), epochs=5)
    # Obtain latent space representation
    autoencoder.eval()
    with torch.no_grad():
        latent_space = autoencoder.encode(data_tensor).numpy()
        latent_df = pd.DataFrame(latent_space, columns=[f'latent_{i}' for i in range(latent_space.shape[1])])

    print(f'Latent space shape: {latent_df.shape}')
    # =======================================================
    # Noise filtering and clustering 
    min_cluster = 30 #in_cluster_calc(frequency=50, eval_window=0.02)
    min_samples = 15
    isolated_df = isolationForest(latent_df)
    
    # Preserve indices for mapping back to original data
    isolated_indices = isolated_df.index
    
    n_components = getNComponents(latent_df)
    clustered_df = hdbscan(isolated_df, n_components=n_components, min_cluster_size=min_cluster, min_samples=min_samples, metric='euclidean')

    # Map back to original data using preserved indices
    # Extract the data that survived isolation forest
    survived_data = data_normalised.loc[isolated_indices].copy()
    survived_features = normalised_features.loc[isolated_indices].copy()
    
    # Reset indices to align everything
    survived_data.reset_index(drop=True, inplace=True)
    survived_features.reset_index(drop=True, inplace=True)
    clustered_df.reset_index(drop=True, inplace=True)
    
    # Add ID column to clustered_df
    if 'id' in survived_data.columns:
        clustered_df['id'] = survived_data['id']
    else:
        print(f"Warning: 'id' column not found in data. Available columns: {survived_data.columns.tolist()}")
        # Create sequential IDs as fallback
        clustered_df['id'] = range(len(clustered_df))
    
    # =======================================================
    # PD Classification using Physics-Informed Features
    print("\n=============== PD CLASSIFICATION ===============")
    
    # Get feature thresholds from domain knowledge
    thresholds = get_feature_thresholds()
    
    # Assign physics-informed weights to features
    weights = assignWeights(thresholds)
    
    # Aggregate features at cluster level
    cluster_features = aggregate_cluster_features(clustered_df, survived_features)
    
    # Compute weighted scores for each cluster
    scores_df = computeScores(cluster_features, thresholds, weights)
    
    # Classify clusters as PD or noise using multi-measure voting
    # Parameters can be tuned based on your system characteristics:
    # - score_threshold: 0.6 means 60% of weighted votes must pass
    # - min_votes: 3 means at least 3 of 5 features must exceed threshold
    classification_df = classify_clusters(scores_df, score_threshold=0.6, min_votes=3)
    
    # Map cluster-level labels back to individual events
    labeled_data = map_labels_to_events(clustered_df, classification_df)
    
    # Optional: Analyze features for debugging/tuning
    # analyze_features(survived_features)
    
    # =======================================================
    # Write Results to db 
    writeResults(labeled_data, classification_df, cfg, configPath)
    
    print("\n=============== AUTOMATED DE-NOISING COMPLETE ===============")
    print(f'Total Time Taken: {(time.time() - start_time):.2f}s\n')

    return 0    


if __name__ == "__main__":
    configPath = pathlib.Path("C:/Users/ldgre/Desktop/wfh/holy-grain/config.json")
    main(configPath)