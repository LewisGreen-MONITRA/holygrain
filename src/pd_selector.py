"""
Multi Measure Ensemble PD selector. 

Combines multiple selection methods to robustly identify partial discharge events from clustered data. 

Thresholds of each measure to be defined either from clustering statistics or domain knowledge.
Likely a combination of both. 

Map cluster assingments to 1 for PD and 0 for noise 

Can then write these results back to database, to be loaded into heatmap viewer. 

"""
