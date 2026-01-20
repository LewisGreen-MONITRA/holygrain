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
import torch
import torch.nn as nn
import torch.nn.functional as F 

from data_preperation import getDataset, getNComponents, normaliseDataset
from autoencoder import PhysicsInformedAutoencoder
from feature_extraction import extract_pd_features, normalise_features, analyze_features 
from clustering import hdbscan, isolationForest, min_cluster_calc

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
    
    # =======================================================
    # Load Configuration and Dataset
    cfg = load_config(configPath)
    print("Configuration loaded successfully:")
    # normalise the raw data to get normal distribution 
    data_normalised, transformers = normaliseDataset(cfg)

    # =======================================================
    # Feature Extraction and Normalisation
    # Domain specific feature extraction, kurtosis etc. 
    features = extract_pd_features(data_normalised)
    normalised_features, domain_transformers = normalise_features(features)


    # =======================================================
    # Autoencoder Initialisation 
    #loader = torch.utils.data.DataLoader(
    #    torch.tensor(normalised_features.values, dtype=torch.float32),
    #    batch_size=32,
    #    shuffle=True,
    #    drop_last=True)
    #autoencoder = PhysicsInformedAutoencoder(signal_length= 11, latent_dim=16)    
    #optimiser = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    # Training 


    # =======================================================
    # Noise filtering and clustering 
    min_cluster = 30 #in_cluster_calc(frequency=50, eval_window=0.02)
    min_samples = 15
    isolated_df = isolationForest(data_normalised)
    n_components = getNComponents(data_normalised)
    clustered_df = hdbscan(isolated_df, n_components=n_components, min_cluster_size=min_cluster, min_samples=min_samples, metric='euclidean')
    print(clustered_df.describe())
    print(features.describe())
    
    return 0    


if __name__ == "__main__":
    configPath = pathlib.Path("C:/Users/lewis.green/Desktop/holy grain/config.json")
    main(configPath)