"""
Helper functions

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def plot_clusters(df):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['observedTime_ms'], df['energy'], c=df['cluster'], cmap='rainbow', s=10)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_title('Clustering Results')
    ax.set_xlabel('Observed Time (ms)')
    ax.set_ylabel('Energy')
    plt.show()


def validate_clusters(df):
    """
    Placeholder for cluster validation metrics.
    Could implement silhouette score, Davies-Bouldin index, etc.
    """
    features = df.select_dtypes(include=[np.number]).drop(columns=['cluster', 'id', 'acquisition_id'], errors='ignore')
    labels = df['cluster'].values

    if len(set(labels)) > 1 and len(features) > 0:
        sil_score = silhouette_score(features, labels)
        db_score = davies_bouldin_score(features, labels)
        ch_score = calinski_harabasz_score(features, labels)

        print(f'Silhouette Score: {sil_score:.4f}')
        print(f'Davies-Bouldin Index: {db_score:.4f}')
        print(f'Calinski-Harabasz Score: {ch_score:.4f}')
    else:
        print('Not enough clusters or features to compute validation metrics.')

def sampleDataset(df, seed):
    """
    Get a subsample of the dataset. 
    Uses Yamane simplified method to determine sample size.

    """
    if isinstance(df, pd.DataFrame ): # DataFrame (not reduced)
         n_samples = len(df.index) / (1 + (len(df.index) * 0.01**2))
         sample = df.sample(round(n_samples), random_state= seed)

    else: # numpy array (pca reduced)
         n_samples =  df[:,0].shape[0] / (1 +  df[:,0].shape[0] * 0.01**2) 
         df = pd.DataFrame(df)
         sample = df.sample(round(n_samples), random_state=seed)

    return sample
