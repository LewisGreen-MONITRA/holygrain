"""
Helper functions

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_clusters(df):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['observedTime_ms'], df['energy'], c=df['cluster'], cmap='tab20', s=10)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_title('Clustering Results')
    ax.set_xlabel('Observed Time (ms)')
    ax.set_ylabel('Energy')
    plt.show()


# TODO add cluster validation metrics 
def validate_clusters(df):
    """
    Placeholder for cluster validation metrics.
    Could implement silhouette score, Davies-Bouldin index, etc.
    """
    pass
