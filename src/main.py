"""
Phyiscs inofrmed unsupervised approach for automated de-noiseing of Partial Discharge signals.

Author: Lewis Green
Date: 2024-06-15
MOINTRA 
"""
import numpy as np
import pandas as pd
import sqlite3
import pathlib 
import json 

from data_preperation import getDataset 
from clustering import dbscan


def load_config(configPath: pathlib.Path):

    with configPath.open() as f: 
        cfg = json.load(f)
   
    # create dictionary of cfg contents. Access these at a later date
    configDict = {
        "databaseFile": cfg.get("databaseFile"),
        "channelNumber": cfg.get('channelNumber'),
        "startTime": cfg.get('startTime'),
        "endTime": cfg.get('endTime')
     }

    # Add explicit checks for critical keys
    if configDict["databaseFile"] is None:
        raise ValueError(f"Configuration file '{configPath}' is missing 'databaseFile' key.")
    if configDict["channelNumber"] is None:
        raise ValueError(f"Configuration file '{configPath}' is missing 'channelNumber' key.")
    if configDict["startTime"] is None:
        raise ValueError(f"Configuration file '{configPath}' is missing 'startTime' key.")
    if configDict["endTime"] is None:
        raise ValueError(f"Configuration file '{configPath}' is missing 'endTime' key.")

    return configDict



def main(configPath: pathlib.Path):
    cfg = load_config(configPath)
    

    return 0    


if __name__ == "__main__":
    configPath = pathlib.Path("C:/Users/lewis.green/Desktop/holy grain/config.json")
    main(configPath)