"""
Clusterting section of unsupervise pipeline. 
DBSCAN to identify inherent outliers for filtering 


"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def calcEpsilon(df, radius= 1.0):
    """
    Determine Epsilon for DBSCAN for a given dataset.
    Estimate by examining the "elbow/knee" of the curve produced by a k-distance plot
    Adapted from Schubert et al 2017
    doi/10.1145/3068335
    Auto detection of the "Knee" with a kneedle algorithm implementation
    Adapted from Satopaa et al 2011
    doi/10.1109/ICDCSW.2011.20
    """

    k = 2 * df.shape[-1] - 1 
    nn = NearestNeighbors(n_neighbors=k, radius=radius)
    nn.fit(df)
    distances, indicies = nn.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:, - 1]

    n_points = len(distances)
    coords = np.vstack((range(n_points), distances)).T
    # convert to 1d array
    np.array([range(n_points), distances])
    first = coords[0]
    line_vec = coords[-1] - coords[0]
    # nomralise
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    # point from first event
    vec_from_first = coords - first
    dot_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(dot_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    # normalise
    dist_to_line = np.linalg.norm(vec_to_line, axis=1)
    # max curvature
    index = np.argmax(dist_to_line)
    # value of k distances at max curvature is the eps value
    eps = distances[index]
    return eps


def dbscan(df, n_componnents):
    """
    Docstring for dbscan
    
    :param df: Description
    :param n_componnents: Description
    """
    min_sampls = len(df.columns) * 2

    pca = PCA(n_components=n_componnents).fit_transform(df)
    eps = calcEpsilon(pca, radius=1.0)
    clf = DBSCAN(eps=eps, min_samples=min_sampls, metric='euclidean', algorithm='kd_tree').fit(pca)
    labels = clf.labels_   
    n_cluster_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f'Epsilon: {eps}')
    print(f'Estimated clustser: {n_cluster_}')
    print(f'Estimated noise points: {n_noise_}')
    df['cluster'] = labels 

    return df 

def gaussianMixture(df):
    """
    Docstring for gaussianMixture
    
    :param df: Description
    """


