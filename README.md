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





#### **Physics Informed Auto-Encoder**

Auto-Encoder incorporates the physical properties of PD signals to determine the latent embeddings of the feature space. 



#### **Clustering**

Extreme Outlier removal performed by an Isolation Forest, followed by Hierarchical-DBSCAN to cluster events with similar properties. Cluster labels are assigned to each event before being passed to ensemble voter. 



#### **Ensemble Voting**

Sensor Sensitive feature weighting informs how each event is classified after cluster assignment.

Supports HFCT, UHF, HVCC and TEV. For each cluster, confidence scores are calculated based on adaptive threshold weightings, voting is then performed. 





### **Data Pipeline**

1. **Config File. Contains path to database, channel number, acquisition start and end time.** 
2. **Data pulled from db to script, normalised, sensor information and event count acquired.** 
3. **PD Features Extracted and normalised.** 
4. **Auto-Encoder trained on normalised data, returns latent embeddings after reconstruction** 
5. **Latent Embeddings passed to isolation forest for initial outlier removal. Returns clean "Isolated" data.** 
6. **Isolated Data passed to HDBSCAN** 
7. **Thresholds and weights calculated based on sensor type, passed onto the voting classifier.**  
8. **Ensemble voting performed, returns labelled classification of pd/noise.**
9. **Results Written back to db, later to be used in Hyperion Heatmap-Viewer to produce PRPD pattern.**



