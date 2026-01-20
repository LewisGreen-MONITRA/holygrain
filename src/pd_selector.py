"""
Multi Measure Ensemble PD selector. 

Combines multiple selection methods to robustly identify partial discharge events from clustered data. 

Thresholds of each measure to be defined either from clustering statistics or domain knowledge.
Likely a combination of both. 

Map cluster assingments to 1 for PD and 0 for noise 

Can then write these results back to database, to be loaded into heatmap viewer. 

"""
import numpy as np
import pandas as pd 
import sqlite3
import json 
import uuid

from sklearn.ensemble import VotingClassifier


def writeResults(df, cfg, path):
  """
  Write Cluster Labels to Cluster db.

  """
  conn = sqlite3.connect(cfg['databaseFile'])
  cursor = conn.cursor()

  cursor.execute("DROP TABLE IF EXISTS Event_Cluster")
  #cursor.execute("DROP TABLE IF EXISTS Process") # drops the process table, want to keep this?
  #cursor.execute("DROP TABLE IF EXISTS Cluster")
  conn.commit()

  cursor.execute("CREATE TABLE Event_Cluster (events_id INTEGER, clusters_id INTEGER, FOREIGN KEY(events_id) REFERENCES Event(id), FOREIGN KEY(clusters_id) REFERENCES Cluster(id))")
  #cursor.execute("CREATE TABLE Process (id INTEGER PRIMARY KEY AUTOINCREMENT, uuidStr TEXT, startTime INTEGER, endTime INTEGER)")
  #cursor.execute("CREATE TABLE Cluster (id INTEGER PRIMARY KEY AUTOINCREMENT, clusterNumber INTEGER, process_id INTEGER NOT NULL, FOREIGN KEY(process_id) REFERENCES Process(id))")
  conn.commit()
  # update the config with a new uuid, this will only occur if the clustering is succesful

  uuidStr = str(uuid.uuid4())
  cfg["processUuid"] = uuidStr
  # write to config json.
  with path.open('w') as f:
    json.dump(cfg, f)

  """
  Process block
  """
  # populate columns
  process_entry = pd.DataFrame({
    'uuidStr': [cfg['processUuid']],
    'startTime': [cfg['startTime']],
    'endTime': [cfg['endTime']]
  })

  process_entry.to_sql("Process", conn, if_exists='append', index=False)
  conn.commit()
  process_df = pd.read_sql_query("SELECT id FROM Process WHERE uuidStr =? ORDER BY id DESC LIMIT 1",  conn, params=(uuidStr,))
  # sanity check
  if process_df.empty:
    raise ValueError(f'Could not retrive processUuid value')

  # assign process id to latest process
  process_id = process_df['id'].iloc[-1].item()

  """
  Cluster block
  """
  clusters = df['cluster'].unique()
  cluster_data = pd.DataFrame({
    'clusterNumber': clusters,
    'process_id': process_id
  })
  # print(cluster_data.head(1)) # cluster number and process id exists
  # print(cluster_data.shape)
  cluster_query = """
  INSERT INTO Cluster (clusterNumber, process_id)
  VALUES (?, ?)
  """
  cursor.executemany(cluster_query, cluster_data.values.tolist())
  conn.commit()
  # or its not pulling the changes? not pulling the changes, can see the commit in dbeaver
  cluster_df = pd.read_sql_query("SELECT id, clusterNumber FROM Cluster WHERE process_id =?", conn, params=(process_id,))
  """
  Event_Cluster block
  """
  # get id associated with cluster number that is associate with an event
  # map it to the cluster id
  cluster_mapping = cluster_df.set_index('clusterNumber')['id'].to_dict()
  clusters_id = df['cluster'].map(cluster_mapping).values

  events_cluster = pd.DataFrame({
    'events_id': df['id'], # events, cluster
    'clusters_id': clusters_id # need to assign a cluster id to each of the event ids
  # 'procecss_id' : process_id
  })

  #print(events_cluster.head())
  events_cluster.to_sql('Event_Cluster', conn, if_exists='append', index=False) # write to db
  conn.commit() # commit transaction
  conn.close() # close, don't need access anymore

  return
