### **MONITRA "Holy-Grain" Automated De-Noising of PD signals**

#### 

#### **Introduction**

Complete end-to-end pipeline for automated differentiation between noise and partial discharge (PD) signals in high voltage systems using physics informed representation learning, clustering and ensemble classification.



#### **PD Feature Extraction**

Key Physical Features of PD signals are extracted from the data for later use to inform weightings of the ensemble voter. 

Features Include

* Kurtosis
* Phase Consistency
* Energy Concentration
* Signal-to-Noise Ratio (SNR)
* Repetition Rate Regularity

#### 

#### **Physics Informed Auto-Encoder**

Auto-Encoder incorporates the physical properties of PD signals to determine the latent embeddings of the feature space. Coefficients of wavelets are used as part of the input features to capture signal characteristics.

#### **Clustering**

Extreme Outlier removal performed by an Isolation Forest, followed by Hierarchical-DBSCAN to cluster events with similar properties. Cluster labels are assigned to each event before being passed to ensemble voter. 

#### **Ensemble Voting**

Sensor Sensitive feature weighting informs how each event is classified after cluster assignment.

Supports HFCT, UHF, HVCC and TEV. For each cluster, confidence scores are calculated based on adaptive threshold weightings, voting is then performed. 

### **Data Pipeline**

1. **Config File. Contains path to database, channel number, acquisition start and end time.** 
2. **Data pulled from db to script, normalised, sensor information and event count acquired.** 
3. **PD Features Extracted and normalised.** 
4. **Auto-Encoder trained on normalised data, returns latent embeddings after reconstruction.** 
5. **Latent Embeddings passed to isolation forest for initial outlier removal. Returns clean "Isolated" data.** 
6. **Isolated Data passed to HDBSCAN, clusters are identified.** 
7. **Thresholds and weights calculated based on sensor type, passed onto the voting classifier.**  
8. **Ensemble voting performed, returns labelled classification per cluster for pd/noise.**
9. **PD/Noise labels mapped back to each event.** 
10. **Results Written back to db, later to be used in Hyperion Heatmap-Viewer to produce PRPD pattern.**

### **Parameter Tuning**

#### **Adaptive weight thresholding**

Weight thresholding is determined by tuning the percentile parameter in the get_adaptive_thresholds function. Lower percentiles lead to more lenient thresholds, classifying more clusters as PD, higher percentiles lead to stricter thresholds, classifying less clusters as PD. Recommend using 50-75th percentiles. 


#### **HDBSCAN Metrics and Pairwise Distance Algorithm**

Current support for kd_tree and ball_tree algorithms for HDBSCAN implementation. L2 Euclidean distance support for pairwise distances, defaults to kd_tree when uses L2 distance

#### **What to do with datasets with high number of events?**
Training times for this pipeline can be high when datasets contain large number of events (>1 million) with runtime taking hours on cpu. Suggest utilsiing when event count ~500k, taking roughly 40 minutes on i5-14500. 
Options to reduce training time include:
* Subsampling dataset to reduce event count and then map the labels back to the full dataset using K-Nearest Neighbors. Although this approach would break the physics informed nature of the pipeline. 
* Utilising GPU acceleration for auto-encoder training and HDBSCAN execution. cuML library has HDBSCAN implementation that can be used to speed up clustering times. Script could then be run on RDS with the nvidia GPU instance.   