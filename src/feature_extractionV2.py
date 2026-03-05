"""
V2 of feature extraction
refactor v1 


- db specific feature extraction functions
- features [observedPhaseDegrees, energy,
peakValue, observedArea_mVns, observedFallTime_ns,
observedPeakWidth_10pc_ns, observedRiseTime_ns, observedTime_ms]

- phase weighting
- energy level weighting
- vectorised the implementations 

"""
import numpy as np
import pandas as pd
from scipy import signal, stats 


def compute_phase_weighting(data, verbose=True): 
    """
    Compute phase weighting for PD events based on proximity to 90 and 270 degrees.
    Closer to these angles => higher weight, as they are more indicative of PD activity. 
    Vectorised implementation for efficiency.

    Args:
        data: numpy array or DataFrame containing 'observedPhaseDegrees' column or index 8 for phase
        verbose: If True, prints progress information.
    Returns:
        Array of phase weights corresponding to each sample in data.
    """
    if isinstance(data, pd.DataFrame):
        phases = data['observedPhaseDegrees'].values
    else:
        # observed phase degree is at the last index after id and acqui id are dropped 
        phases = data[:, -1]
    
    if verbose:
        print("Computing phase weighting...")
    # stronger weighting for events near 90 and 270 degrees +- 45 degrees as a range 
    if isinstance(data, pd.DataFrame):
        weights = np.where(
            ((phases >= 45) & (phases <= 135)) | ((phases >= 225) & (phases <= 315)),
            # normalise the range so that the max weight is 1 at 90 and 270 degrees, and decreases linearly to above 0 at the edges of the range
            1 - (np.abs(phases - 90) / 45) * (phases <= 135) - (np.abs(phases - 270) / 45) * (phases >= 225),
            0.5  # baseline weight for events outside the target phase ranges
        )
        return weights




def compute_energy_weighting(data, verbose=True):
    """
    Compute energy weighting for PD events based on the 'energy' feature.
    Higher energy => higher weight, as they are more likely to be authentic PD events. 
    Events with energy near the noise floor should receive lower weights.
    Vectorised implementation for efficiency.

    Args:
        data: numpy array or DataFrame containing 'energy' column or index 1 for energy
        verbose: If True, prints progress information.
    Returns:
        Array of energy weights corresponding to each sample in data.
    """ 
    if isinstance(data, pd.DataFrame):
        energy = data['energy'].values
    else:
        energy = data[:, 1]  # energy is at index 1
    
    if verbose:
        print("Computing energy weighting...")
    

    return 
