"""
Clusterting section of unsupervise pipeline. 
DBSCAN to identify inherent outliers for filtering 


"""

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest 


def isolationForest():
    """
    Isolation forest acting as a pre-filter before dbscan clustering. 
    Should result in a more robust clustering outcome

    Input should be the latent space produced by autoencoder. 
    Return filtered dataframe and scores

    """



def hdbscan(df, n_components):
    """
    In v1 of clustering, dbscan was implemented.
    Switch to HDBSCAN as this can leverage the inherent charactersitics of PD signals

    Input should be the filtered dataframe from isolation forest. 
    Returns cluster assignments for each event. 

    """

    pca = PCA(n_components=n_components).fit_transform(df)


    clf = HDBSCAN()
    clf.fit(pca)

    labels = clf.labels_ 

