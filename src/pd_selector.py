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

def aggregate_cluster_features(clustered_df, features):
   """
   Compute statistics of features at the cluster level.
   
   :param clustered_df: DataFrame with cluster assignments (must have 'cluster' column)
   :param features: DataFrame of features extracted from feature extraction step
   :returns: DataFrame with cluster-level statistics for each feature
   """
   # Reset indices to ensure alignment
   clustered_reset = clustered_df[['cluster']].reset_index(drop=True)
   features_reset = features.reset_index(drop=True)
   
   combined = pd.concat([clustered_reset, features_reset], axis=1)
   
   # Compute statistics per cluster
   cluster_stats = combined.groupby('cluster').agg({
        'kurtosis': ['mean', 'std', 'median'],
        'phase_consistency': ['mean', 'std', 'median'],
        'energy_concentration': ['mean', 'std', 'median'],
        'snr': ['mean', 'std', 'median'],
        'repetition_regularity': ['mean', 'std', 'median']
    })
   
   # Flatten column names
   cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
   
   # Add cluster size
   cluster_stats['cluster_size'] = combined.groupby('cluster').size()
   
   print(f'Aggregated features for {len(cluster_stats)} clusters')

   return cluster_stats 

def assignWeights(feature_thresholds):
    """
    Define weights based on feature thresholds.
    Higher threshold = lower weight (harder to achieve).
    
    :param feature_thresholds: Dict of feature name -> threshold value
    :returns: Dict of normalized weights summing to 1.0
    """
    weights =  {}
    # Assign inverse threshold as base weight (lower threshold = lower weight)
    for feature, threshold in feature_thresholds.items():
        if threshold > 0:
            weights[feature] = 1.0 / threshold 
        else:
            weights[feature] = 1.0
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())  
    weights = {k: v / total_weight for k, v in weights.items()}
    
    print("Feature weights assigned:")
    for feature, weight in weights.items():
        print(f'  {feature}: {weight:.4f}')

    return weights

def computeScores(cluster_stats, feature_thresholds, weights=None):
   """
   Compute weighted scores for each cluster based on feature thresholds.
   
   :param cluster_stats: DataFrame from aggregate_cluster_features()
   :param feature_thresholds: Dict of feature name -> threshold value
   :param weights: Optional dict of feature weights (default: equal weights)
   :returns: DataFrame with cluster scores and votes
   """
   if weights is None:
      weights = {feature: 1.0 / len(feature_thresholds) for feature in feature_thresholds.keys()}
   
   scores = []

   for cluster_id, row in cluster_stats.iterrows():
       cluster_score = {
          'cluster': cluster_id,
          'cluster_size': row.get('cluster_size', 0),
          'weighted_score': 0.0,
          'votes_passed': 0
       }

       # Check each feature against threshold
       for feature, threshold in feature_thresholds.items():
            # Use mean value for comparison
            feature_mean_col = f'{feature}_mean'

            if feature_mean_col in row.index:
                feature_value = row[feature_mean_col]

                # Vote: 1 if above threshold, 0 otherwise
                vote = 1 if feature_value >= threshold else 0
                cluster_score[f'{feature}_vote'] = vote
                cluster_score[f'{feature}_value'] = feature_value

                # Add weighted contribution
                if vote == 1:
                    cluster_score['weighted_score'] += weights[feature]
                    cluster_score['votes_passed'] += 1

       scores.append(cluster_score)
   
   scores_df = pd.DataFrame(scores)
   scores_df = scores_df.set_index('cluster')
   
   print(f'Computed scores for {len(scores_df)} clusters')
   print(f'  Avg weighted score: {scores_df["weighted_score"].mean():.3f}')
   print(f'  Avg votes passed: {scores_df["votes_passed"].mean():.1f}/5')

   return scores_df

def classify_clusters(scores_df, score_threshold=0.35, min_votes=2):
    """
    Classify clusters as PD or noise based on score and vote thresholds.
    
    :param scores_df: DataFrame from computeScores()
    :param score_threshold: Minimum weighted score to classify as PD
    :param min_votes: Minimum number of feature votes to classify as PD
    :returns: DataFrame with cluster classifications
    """
    classification = []

    for cluster_id, row in scores_df.iterrows():
        # Classify as PD if BOTH conditions met:
        # 1. Weighted score >= threshold
        # 2. At least min_votes features passed their thresholds
        is_pd = 1 if (row['weighted_score'] >= score_threshold and row['votes_passed'] >= min_votes) else 0
        
        classification.append({
            'cluster': cluster_id,
            'is_pd': is_pd,
            'weighted_score': row['weighted_score'],
            'votes_passed': row['votes_passed'],
            'cluster_size': row.get('cluster_size', 0)
        })
    
    classification_df = pd.DataFrame(classification)

    n_pd_clusters = (classification_df['is_pd'] == 1).sum()
    n_noise_clusters = (classification_df['is_pd'] == 0).sum()
    
    # Calculate event counts
    pd_events = classification_df[classification_df['is_pd'] == 1]['cluster_size'].sum()
    noise_events = classification_df[classification_df['is_pd'] == 0]['cluster_size'].sum()

    print(f'\nClassification Results:')
    print(f'  PD clusters: {n_pd_clusters} ({pd_events} events)')
    print(f'  Noise clusters: {n_noise_clusters} ({noise_events} events)')

    return classification_df

def map_labels_to_events(clustered_df, classification_df):
    """
    Map cluster level classification back to event level.
    
    :param clustered_df: Dataframe with cluster assignments
    :param classification_df: Dataframe with cluster level classification
    """
    # mapping dict
    label_mapping = classification_df.set_index('cluster')['is_pd'].to_dict()

    labeled_data = clustered_df.copy()
    labeled_data['is_pd'] = labeled_data['cluster'].map(label_mapping)
    
    labeled_data['is_pd'].fillna(0, inplace=True)  # Treat unmapped clusters as non-PD
    labeled_data['is_pd'] = labeled_data['is_pd'].astype(int)

    n_pd_events = (labeled_data['is_pd'] == 1).sum()
    n_noise_events = (labeled_data['is_pd'] == 0).sum()

    print(f'Mapped to events: {n_pd_events} PD, {n_noise_events} noise')

    return labeled_data

def subtype_classification(cluster_stats):
    """
    Optional: Further classify PD events into subtypes based on feature patterns.
    should be a physics informed appraoched rather than using pure ml 
    done at the cluster level then mapped back to events to maintain schema integrity. 


    
    :param cluster_stats: DataFrame with cluster level statistics
    :returns: DataFrame with additional subtype classifications
    """
    # Placeholder for subtype classification logic
    # For example, could classify PD events into  slot discharge, corona, etc. based on feature thresholds
    # Potentially utilise hidden markov models to model pd types as latent regimes in the data. 
    # produce a prediction of pd type with conficdence score 
    # create a physics rule engine to interpret these predictions into subtype classifications.
    # rules for each of the subtypes to be defined based on domain knowledge 
    # for each cluster 
    # label if pd or not 
    # if pd, apply rules and create confidence score for each subtype classification
    # highest score -> aggisn subtype label to cluster
    # map back to events as per previous function
    # if not pd then subtype = 'noise' 

    # need to define a feature set at the cluster level 
    # so aggregate features from feature extraction step to cluster level
    # can pull those stats straight from the cluster_stats dataframe produced in the previous step. 
    # then define thresholds for each subtype classification based on these features.
    
    

    # set thresholds for each subtype 

    internal = { # internal delamination 
       
       
    }
    
    surface = { # slot discharge etc. 
    
    }

    corona = {
       
       
    }

    floating = { 
       
    }

    

    return


def writeResults(df, classification_df, cfg, path):    
  """
  Write Cluster Labels to Cluster db.
  
  Maps PD classification to cluster numbers to maintain schema integrity.
  
  :param df: DataFrame with cluster assignments and event ids
  :param classification_df: DataFrame with cluster-level PD classification (cluster, is_pd)
  :param cfg: Configuration dictionary
  :param path: Path to config file for updating UUID
  """
  conn = sqlite3.connect(cfg['databaseFile'])
  cursor = conn.cursor()

  # Drop and recreate tables
  cursor.execute("DROP TABLE IF EXISTS Event_Cluster")
  conn.commit()

  cursor.execute("""
    CREATE TABLE Event_Cluster (
      events_id INTEGER, 
      clusters_id INTEGER, 
      FOREIGN KEY(events_id) REFERENCES Event(id), 
      FOREIGN KEY(clusters_id) REFERENCES Cluster(id)
    )
  """)
  conn.commit()
  
  # Update the config with a new uuid
  uuidStr = str(uuid.uuid4())
  cfg["processUuid"] = uuidStr
  with path.open('w') as f:
    json.dump(cfg, f, indent=2)

  """
  Process block
  """
  process_entry = pd.DataFrame({
    'uuidStr': [cfg['processUuid']],
    'startTime': [cfg['startTime']],
    'endTime': [cfg['endTime']]
  })

  process_entry.to_sql("Process", conn, if_exists='append', index=False)
  conn.commit()
  process_df = pd.read_sql_query(
    "SELECT id FROM Process WHERE uuidStr =? ORDER BY id DESC LIMIT 1",  
    conn, params=(uuidStr,)
  )
  
  if process_df.empty:
    raise ValueError(f'Could not retrieve processUuid value')

  process_id = process_df['id'].iloc[-1].item()

  """
  Cluster block - now includes is_pd classification
  """
  clusters = df['cluster'].unique()
  
  # Create mapping from cluster number to is_pd classification
  if classification_df is not None and not classification_df.empty:
    pd_mapping = classification_df.set_index('cluster')['is_pd'].to_dict()
  else:
    # Default: no clusters are PD if no classification provided
    pd_mapping = {c: 0 for c in clusters}
  
  # Build cluster data with is_pd flag
  cluster_data = pd.DataFrame({
    'clusterNumber': clusters,
    'process_id': process_id,
    'is_pd': [pd_mapping.get(c, 0) for c in clusters]  # Default to noise if not found
  })

  # Ensure Cluster table has is_pd column
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS Cluster (
      id INTEGER PRIMARY KEY AUTOINCREMENT, 
      clusterNumber INTEGER, 
      process_id INTEGER NOT NULL,
      is_pd INTEGER DEFAULT 0,
      FOREIGN KEY(process_id) REFERENCES Process(id)
    )
  """)
  conn.commit()
  
  cluster_query = """
  INSERT INTO Cluster (clusterNumber, process_id)
  VALUES (?, ?)
  """
  cursor.executemany(cluster_query, cluster_data[['clusterNumber', 'process_id']].values.tolist())
  conn.commit()
  
  cluster_df = pd.read_sql_query(
    "SELECT id, clusterNumber FROM Cluster WHERE process_id =?", 
    conn, params=(process_id,)
  )
  
  """
  Event_Cluster block
  """
  cluster_mapping = cluster_df.set_index('clusterNumber')['id'].to_dict()
  clusters_id = df['cluster'].map(cluster_mapping).values

  events_id = df['id']
  if isinstance(events_id, pd.DataFrame):
    events_id = events_id.iloc[:, 0]
  elif isinstance(events_id, np.ndarray) and events_id.ndim > 1:
    events_id = events_id[:, 0]

  events_cluster = pd.DataFrame({
    'events_id': events_id,
    'clusters_id': clusters_id
  })

  events_cluster.to_sql('Event_Cluster', conn, if_exists='append', index=False)
  conn.commit()
  conn.close()
  
  # Summary
  n_pd_clusters = (cluster_data['is_pd'] == 1).sum()
  n_noise_clusters = (cluster_data['is_pd'] == 0).sum()
  print(f'Wrote {len(clusters)} clusters to database: {n_pd_clusters} PD, {n_noise_clusters} noise')
  print(f'Mapped {len(events_cluster)} events to clusters')

  return
