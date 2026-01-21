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

    """
    clf = IsolationForest(contamination=0.05, random_state = seed)

    clf.fit(df)
    pred = clf.predict(df)

    mask = pred == 1 
    n_outliers = len(mask) - mask.sum()

    cleaned_df = df[mask].copy()
    print(f'Isolation Forest removed {n_outliers} outliers from dataset of size {len(df)}.')
    return cleaned_df 
    

    


def min_cluster_calc(frequency=50, eval_window=0.02):
    """
    Physics informed appraoch to calculating min_cluster_size for hdbscan. 
    Leverage known characteristics of PD signals to inform this choice. 
    E.g., known event rates, expected cluster densities, etc.

    Decide the evaluation window, 50Hz system @ 10 seconds = 500 cycles? 
    Assume weakest plausible pd rate, estimate minimum pulses/cycle 
    Multiply by the cycles 0.2 ppc => surface discharges. 500 cycles => 100 events, forms the lower bound
    Apply saftey margin ~3x should filter switching spikes but retain pd activity.
    300 events minimum cluster size? 

    
    frqeuency: System frequency in Hz 50 or 60 (usa) default 50 
    eval_window: Evaluation window in seconds

    Returns optimal min_cluster_size value.
    
    """

    estimate_pdpc = 0.2 # conservative estimate for pulses per cycle
    cycles = frequency * eval_window 
    min_events = estimate_pdpc * cycles

    safety_margin = 3

    min_cluster_size = int(min_events * safety_margin)
    
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

    pca = PCA(n_components=n_components).fit_transform(df)
    clf = HDBSCAN(min_cluster_size=min_cluster_size,
                   min_samples=min_samples,
                     metric=metric, 
                     algorithm='kd_tree',
                     n_jobs=-1)
    clf.fit(pca)

    labels = clf.labels_ 
    probabilities = clf.probabilities_
    unique_labels = set(labels)

    print(f'Detected {len(unique_labels) - (1 if -1 in labels else 0)} clusters.')
    print(f'Noise points: {(labels == -1).sum()} out of {len(labels)} total points.')
    print(f'Cluster probabilities: min={probabilities.min():.4f}, max={probabilities.max():.4f}, mean={probabilities.mean():.4f}')

    df['cluster'] = labels 
    
    return  df 
