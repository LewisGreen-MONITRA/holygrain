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


def aggregate_cluster_features(clustered_data, features_df):
    """
    Aggregate physics-informed features at the cluster level.
    
    For each cluster, compute statistics (mean, std, median) of the
    5 physics-informed features extracted from feature_extraction module.
    
    Args:
        clustered_data: DataFrame with 'cluster' column (cluster assignments)
        features_df: DataFrame with physics features (kurtosis, phase_consistency, 
                     energy_concentration, snr, repetition_regularity)
    
    Returns:
        cluster_features: DataFrame with aggregated features per cluster
                         Index: cluster_id
                         Columns: mean_<feature>, std_<feature>, count
    """
    # Combine cluster labels with features
    combined = clustered_data[['cluster']].copy()
    combined = pd.concat([combined.reset_index(drop=True), 
                         features_df.reset_index(drop=True)], axis=1)
    
    # Group by cluster and compute statistics
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
    
    print(f"\nCluster feature aggregation complete: {len(cluster_stats)} clusters")
    print(f"Cluster sizes: min={cluster_stats['cluster_size'].min()}, "
          f"max={cluster_stats['cluster_size'].max()}, "
          f"mean={cluster_stats['cluster_size'].mean():.1f}")
    
    return cluster_stats


def assignWeights(feature_thresholds):
    """
    Assign physics-informed weights to each feature based on thresholds.
    
    Features with lower thresholds (easier to pass) get lower weights,
    while features with higher thresholds (harder to pass) get higher weights.
    
    Args:
        feature_thresholds: Dictionary of {feature_name: threshold_value}
    
    Returns:
        weights: Dictionary of {feature_name: weight}
                Normalized to sum to 1.0 for interpretability
    """
    weights = {}
    
    # Assign inverse threshold as base weight (lower threshold = lower weight)
    for feature, threshold in feature_thresholds.items():
        if threshold > 0:
            weights[feature] = 1.0 / threshold
        else:
            weights[feature] = 1.0
    
    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    print(f"\nPhysics-informed feature weights:")
    for feature, weight in weights.items():
        print(f"  {feature:25s}: {weight:.4f}")
    
    return weights


def computeScores(cluster_features, feature_thresholds, weights=None):
    """
    Compute weighted PD scores for each cluster using physics-informed features.
    
    For each cluster, compares aggregated features against thresholds
    and computes a weighted score indicating PD likelihood.
    
    Args:
        cluster_features: DataFrame from aggregate_cluster_features()
                         with mean/std/median for each feature
        feature_thresholds: Dictionary of threshold values per feature
        weights: Optional dictionary of feature weights (default: uniform)
    
    Returns:
        scores_df: DataFrame with columns:
                  - cluster_id
                  - weighted_score (0-1, higher = more PD-like)
                  - votes_passed (number of features above threshold)
                  - individual feature votes
    """
    if weights is None:
        # Uniform weights if not provided
        weights = {feature: 1.0 / len(feature_thresholds) 
                  for feature in feature_thresholds.keys()}
    
    scores = []
    
    for cluster_id, row in cluster_features.iterrows():
        cluster_score = {
            'cluster_id': cluster_id,
            'cluster_size': row['cluster_size'],
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
    
    print(f"\nCluster scoring complete:")
    print(f"  Weighted scores: min={scores_df['weighted_score'].min():.3f}, "
          f"max={scores_df['weighted_score'].max():.3f}, "
          f"mean={scores_df['weighted_score'].mean():.3f}")
    print(f"  Average votes passed: {scores_df['votes_passed'].mean():.1f}/5")
    
    return scores_df


def classify_clusters(scores_df, score_threshold=0.6, min_votes=3):
    """
    Classify clusters as PD (1) or Noise (0) using multi-measure voting.
    
    Uses both weighted score and minimum vote requirements to make
    a robust classification decision.
    
    Args:
        scores_df: DataFrame from computeScores()
        score_threshold: Minimum weighted score for PD classification (0-1)
                        Default: 0.6 (60% of max possible score)
        min_votes: Minimum number of individual feature votes required
                   Default: 3 (majority of 5 features)
    
    Returns:
        classification_df: DataFrame with cluster_id and pd_label (0 or 1)
                          Also includes confidence metrics
    """
    classification = scores_df[['cluster_id', 'cluster_size', 
                                'weighted_score', 'votes_passed']].copy()
    
    # Apply classification rules
    classification['pd_label'] = (
        (classification['weighted_score'] >= score_threshold) & 
        (classification['votes_passed'] >= min_votes)
    ).astype(int)
    
    # Compute confidence (distance from decision boundary)
    classification['confidence'] = np.abs(
        classification['weighted_score'] - score_threshold
    )
    
    # Summary statistics
    n_pd = (classification['pd_label'] == 1).sum()
    n_noise = (classification['pd_label'] == 0).sum()
    total_pd_events = classification[classification['pd_label'] == 1]['cluster_size'].sum()
    total_noise_events = classification[classification['pd_label'] == 0]['cluster_size'].sum()
    
    print(f"\nCluster Classification Results:")
    print(f"  PD clusters: {n_pd} ({n_pd/(n_pd+n_noise)*100:.1f}%)")
    print(f"  Noise clusters: {n_noise} ({n_noise/(n_pd+n_noise)*100:.1f}%)")
    print(f"  PD events: {total_pd_events}")
    print(f"  Noise events: {total_noise_events}")
    print(f"  Average confidence: {classification['confidence'].mean():.3f}")
    
    return classification


def map_labels_to_events(clustered_data, classification_df):
    """
    Map cluster-level PD/noise labels back to individual events.
    
    Args:
        clustered_data: DataFrame with 'cluster' column
        classification_df: DataFrame with cluster_id and pd_label
    
    Returns:
        labeled_data: Original data with added 'pd_label' column
    """
    # Create mapping dictionary
    label_mapping = classification_df.set_index('cluster_id')['pd_label'].to_dict()
    
    # Map labels to events
    labeled_data = clustered_data.copy()
    labeled_data['pd_label'] = labeled_data['cluster'].map(label_mapping)
    
    # Handle any unmapped clusters (e.g., noise cluster -1 from HDBSCAN)
    labeled_data['pd_label'].fillna(0, inplace=True)
    labeled_data['pd_label'] = labeled_data['pd_label'].astype(int)
    
    n_pd_events = (labeled_data['pd_label'] == 1).sum()
    n_noise_events = (labeled_data['pd_label'] == 0).sum()
    
    print(f"\nEvent-level labeling complete:")
    print(f"  PD events: {n_pd_events} ({n_pd_events/len(labeled_data)*100:.1f}%)")
    print(f"  Noise events: {n_noise_events} ({n_noise_events/len(labeled_data)*100:.1f}%)")
    
    return labeled_data 


def writeResults(labeled_data, classification_df, cfg, path):    
  """
  Write Cluster Labels and PD classifications to database.
  
  Args:
    labeled_data: DataFrame with event-level data including 'cluster' and 'pd_label'
    classification_df: DataFrame with cluster-level classifications
    cfg: Configuration dictionary
    path: Path to config file
  """
  conn = sqlite3.connect(cfg['databaseFile'])
  cursor = conn.cursor()

  # Drop and recreate tables
  cursor.execute("DROP TABLE IF EXISTS Event_Cluster")
  cursor.execute("DROP TABLE IF EXISTS Cluster_Classification")
  conn.commit()

  cursor.execute("""
    CREATE TABLE Event_Cluster (
      events_id INTEGER, 
      clusters_id INTEGER, 
      pd_label INTEGER,
      FOREIGN KEY(events_id) REFERENCES Event(id), 
      FOREIGN KEY(clusters_id) REFERENCES Cluster(id)
    )
  """)
  
  cursor.execute("""
    CREATE TABLE Cluster_Classification (
      cluster_id INTEGER PRIMARY KEY,
      cluster_number INTEGER,
      pd_label INTEGER,
      weighted_score REAL,
      votes_passed INTEGER,
      confidence REAL,
      cluster_size INTEGER,
      process_id INTEGER NOT NULL,
      FOREIGN KEY(process_id) REFERENCES Process(id)
    )
  """)
  conn.commit()
  
  # Update config with new UUID
  uuidStr = str(uuid.uuid4())
  cfg["processUuid"] = uuidStr
  with path.open('w') as f:
    json.dump(cfg, f, indent=2)

  # ========================================================================
  # Process Block
  # ========================================================================
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

  # ========================================================================
  # Cluster Block
  # ========================================================================
  clusters = labeled_data['cluster'].unique()
  cluster_data = pd.DataFrame({
    'clusterNumber': clusters,
    'process_id': process_id
  })
  
  cluster_query = """
  INSERT INTO Cluster (clusterNumber, process_id)
  VALUES (?, ?)
  """
  cursor.executemany(cluster_query, cluster_data.values.tolist())
  conn.commit()
  
  cluster_df = pd.read_sql_query(
    "SELECT id, clusterNumber FROM Cluster WHERE process_id =?", 
    conn, params=(process_id,)
  )
  
  # ========================================================================
  # Cluster Classification Block (NEW)
  # ========================================================================
  # Map cluster numbers to IDs
  cluster_id_mapping = cluster_df.set_index('clusterNumber')['id'].to_dict()
  
  classification_export = classification_df.copy()
  classification_export['db_cluster_id'] = classification_export['cluster_id'].map(cluster_id_mapping)
  classification_export['process_id'] = process_id
  
  classification_export[['db_cluster_id', 'cluster_id', 'pd_label', 'weighted_score', 
                         'votes_passed', 'confidence', 'cluster_size', 'process_id']].rename(
    columns={'db_cluster_id': 'cluster_id', 'cluster_id': 'cluster_number'}
  ).to_sql('Cluster_Classification', conn, if_exists='append', index=False)
  conn.commit()
  
  # ========================================================================
  # Event_Cluster Block
  # ========================================================================
  cluster_mapping = cluster_df.set_index('clusterNumber')['id'].to_dict()
  clusters_id = labeled_data['cluster'].map(cluster_mapping).values

  events_cluster = pd.DataFrame({
    'events_id': labeled_data['id'],
    'clusters_id': clusters_id,
    'pd_label': labeled_data['pd_label']
  })

  events_cluster.to_sql('Event_Cluster', conn, if_exists='append', index=False)
  conn.commit()
  conn.close()
  
  print(f"\nResults written to database: {cfg['databaseFile']}")
  print(f"  Process UUID: {uuidStr}")
  print(f"  Process ID: {process_id}")
  print(f"  Tables updated: Event_Cluster, Cluster_Classification, Process, Cluster")

  return
