"""
Clusterting section of unsupervise pipeline. 
DBSCAN to identify inherent outliers for filtering 


"""

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest 

seed = 42 

def isolationForest(df):
    """
    Isolation forest acting as a pre-filter before dbscan clustering. 
    Should result in a more robust clustering outcome

    Input should be the latent space produced by autoencoder. 
    Return filtered dataframe and scores
    3% contamination assumed, don't want to remove authentic pd data 
    """
    clf = IsolationForest(contamination=0.03, random_state = seed)

    clf.fit(df)
    pred = clf.predict(df)

    mask = pred == 1 
   
    n_outliers = len(mask) - mask.sum()
    
    cleaned_df = df[mask].copy()

    print(f'Isolation Forest removed {n_outliers} outliers from dataset of size {len(df)}.')
    return cleaned_df 
    

def min_cluster_calc(acqui_df, cfg , frequency=50):
    """
    Physics informed appraoch to calculating min_cluster_size for hdbscan. 
    Leverage known characteristics of PD signals to inform this choice. 
    E.g., known event rates, expected cluster densities, etc.

    Decide the evaluation window, 50Hz system @ 10 seconds = 500 cycles? 
    Assume weakest plausible pd rate, estimate minimum pulses/cycle 
    Multiply by the cycles 0.2 ppc => surface discharges. 500 cycles => 100 events, forms the lower bound
    Apply saftey margin ~3x should filter switching spikes but retain pd activity.
    
    frqeuency: System frequency in Hz 50 or 60 (usa) default 50 
    eval_window: Evaluation window in seconds

    Returns optimal min_cluster_size value.
    
    """
    
    start_time = cfg['startTime']
    end_time = cfg['endTime']

    eval_window = (end_time - start_time) / 1000  # ms to s
    
    cycles = frequency * eval_window  # total cycles in eval window
    
    avg_events = acqui_df['eventCount'].sum() // len(acqui_df)  # average events per acquisition

    safety_margin = 3

    min_cluster_size = int(avg_events * safety_margin) # lower bound estimate 
    
    return min_cluster_size
    

def hdbscan(df, n_components, min_cluster_size, min_samples, metric='euclidean'):
    """
    In v1 of clustering, dbscan was implemented.
    Switch to HDBSCAN as this can leverage the inherent charactersitics of PD signals

    Input should be the filtered dataframe from isolation forest. 
    n_components for pca
    min cluster size from physics informed calc function. 
    Returns cluster assignments for each event. 

    """

    print(f'Running HDBSCAN clustering with min_cluster_size={min_cluster_size} and PCA n_components={n_components}...')
    
    # Ensure float32 for faster CPU math and lower memory use
    df = df.astype(np.float32, copy=False)

    n_samples = len(df)
    pca_solver = 'randomized' if n_samples > 10000 else 'full'
    print(f'  Processing {n_samples} samples with {pca_solver} PCA...')
    
    pca_transformer = PCA(n_components=n_components, svd_solver=pca_solver, random_state=seed)
    pca = pca_transformer.fit_transform(df)
    
    # Optimized HDBSCAN configuration for sklearn
    clf = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        algorithm='kd_tree' if metric == 'euclidean' else 'ball_tree',  # kd_tree faster for euclidean
        leaf_size=40,                # Tuned for performance
        n_jobs=-1,                   # Parallel processing
        store_centers='centroid',    # Store cluster centers for analysis
    )
    clf.fit(pca)

    labels = clf.labels_ 
    probabilities = clf.probabilities_
    unique_labels = set(labels)

    print(f'Detected {len(unique_labels) - (1 if -1 in labels else 0)} clusters.')
    print(f'Noise points: {(labels == -1).sum()} out of {len(labels)} total points.')
    print(f'Cluster probabilities: min={probabilities.min():.4f}, max={probabilities.max():.4f}, mean={probabilities.mean():.4f}')

    df['cluster'] = labels 
    
    return df 
