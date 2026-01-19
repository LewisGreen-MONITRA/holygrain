"""
Import and clean data, capture data from database as specified in config file 

"""

import sqlite3
import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA

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


def getNComponents(df): 
    """
    Estimated the number of components needed to explain 95% of variance within any given dataset
    Pass number of componets to PCA.

    :param df: Description
    """

    pca = PCA(None).fit(df)

    y = np.cumsum(pca.explained_variance_ratio_)
    x = np.arange(1, len(y) + 1) 

    if (y >= 0.95).any():
        n_components = x[y >= 0.95][0]
    return n_components