"""
Import and clean data, capture data from database as specified in config file 

"""

import sqlite3
import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

seed = 42 


def getDataset(args):
    """
    Recursively find data within a given directory provided by cnf file
    Handle different file types just incase
    Returns unmodified dataset.

    """
    dbPath = args['databaseFile']
    conn = sqlite3.connect(dbPath)
    params = {
        'startTime': args['startTime'],
        'endTime': args['endTime'],
        'channelNumber': args['channelNumber']
    }
    query = """
    SELECT *
    FROM Event AS e
    INNER JOIN Acquisition AS a
    ON a.id = e.acquisition_id
    INNER JOIN Source as s
    ON s.id = a.source_id
    INNER JOIN Channel AS c
    on c.id = s.channel_id
    WHERE c.channelNumber = :channelNumber
    AND a.timestamp >= :startTime
    AND a.timestamp <= :endTime
    """
    try:
        events_df = pd.read_sql_query(query, conn, params=params) # get events data from db
    except sqlite3.OperationalError as e:
        raise ValueError(f"Table missing for query") from e
    else:
    # check that the channel number in the config is the same as what's in the database
    # otherwise creates an empty dataframe and errors out later in the script.
        if events_df.empty:
            # Check if the channelNumber exists in the Channel table at all
            channel_check_query = "SELECT EXISTS(SELECT 1 FROM Channel WHERE channelNumber = :channelNumber)"
            channel_exists = pd.read_sql_query(channel_check_query, conn, params={'channelNumber': args['channelNumber']}).iloc[0, 0]
            if not channel_exists:
                print(f"Warning: Channel number {args['channelNumber']} not found in the database.")
                raise ValueError("Mismatch Between channel number in config and that found in database...")
            else:
                print(f"Warning: No events found for channel {args['channelNumber']} within the specified time range ({args['startTime']} - {args['endTime']}).")
            print(f"Loaded data from {args['databaseFile']}. With Shape: {events_df.shape}")
        else:
            print(f"Loaded data from {args['databaseFile']}. With Shape: {events_df.shape}")
    return events_df

def normaliseDataset(args, seed=42):
    """
    Extends get_dataset.
    Returns a dataframe with a normal distribution.
    Pass to reduced_kmeans_function.
    
    Args:
        args: Configuration dictionary
        seed: Random seed for reproducibility (default: 42)
    """

    try:
        events_df = getDataset(args)
        #features = ['id', 'acquisition_id'] # save for later
        #events_df.drop(features)
        features = ['id', 'energy', 'modifiedFrequency', 'observedArea_mVns', 'observedFallTime_ns',
                    'observedPeakWidth_10pc_ns', 'observedPhaseDegrees',
                    'observedRiseTime_ns',  'observedTime_ms', 'peakValue', 'acquisition_id']

        transformers = {}

        reduced_df = events_df.copy()
        reduced_df = reduced_df[features]

        cols = [col for col in reduced_df.columns if col != 'id']
        for col in cols:
            transformer = QuantileTransformer(output_distribution="normal", random_state=seed)
            vec_len = len(reduced_df[col].values)
            raw_vec = reduced_df[col].values.reshape(vec_len, 1)
            transformer.fit(raw_vec)
            transformers[col] = transformer
            reduced_df[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]

        # rename ids to correspond with the tables that they came from
        # change names after transofmer has run through
        # Note: The features list already has correct names, so renaming is only needed if columns differ
        if len(reduced_df.columns) == 11:
            # Columns already match feature count, ensure correct order
            reduced_df.columns = ['id', 'energy', 'modifiedFrequency', 'observedArea_mVns', 'observedFallTime_ns',
                                 'observedPeakWidth_10pc_ns', 'observedPhaseDegrees',
                                 'observedRiseTime_ns',  'observedTime_ms', 'peakValue', 'acquisition_id']
        # If column count differs, keep original column names
        return reduced_df, transformers

    except FileNotFoundError:
        print(f"ERROR: Failed to Find Event Data at {args['databaseFile']}")
        print(f"Check File Exists or Update the Path")
        raise

def getSensor(args):
    """
    Get sensor type from database
    args: Config dictionary 

    Returns sensor type as a string.
    """

    query = """
    SELECT sensorType
    FROM Channel
    WHERE channelNumber = :channelNumber
    """

    channelDdf = pd.read_sql_query(query, sqlite3.connect(args['databaseFile']),
                                    params={'channelNumber': args['channelNumber']})
    sensor = channelDdf['sensorType'].values
    
    return sensor

def getEventCount(cfg):
    query = """
    SELECT eventCount 
    FROM Acquisition AS a
    """
    acqui_df = pd.read_sql_query(query, sqlite3.connect(cfg['databaseFile']))
    return acqui_df

def inverseTransform(reduced_df, transformers):
    """
    Recover original data structure

    """
    df = reduced_df.copy()
    cols = [col for col in df.columns if col in transformers]

    for col in cols:
        transformer = transformers[col]
        norm_vec = df[col].values.reshape(-1, 1)
        df[col] = transformer.inverse_transform(norm_vec).flatten()

    return df


def getNComponents(df): 
    """
    Estimated the number of components needed to explain 95% of variance within any given dataset
    Pass number of componets to PCA.
    Uses randomized PCA for faster computation on large datasets.

    :param df: Description
    """
    n_samples = len(df)
    # Use randomized SVD solver for faster computation on large datasets
    pca_solver = 'randomized' if n_samples > 10000 else 'full'
    pca = PCA(n_components=None, svd_solver=pca_solver, random_state=seed).fit(df)

    y = np.cumsum(pca.explained_variance_ratio_)
    x = np.arange(1, len(y) + 1) 

    if (y >= 0.95).any():
        n_components = x[y >= 0.95][0]
    return n_components