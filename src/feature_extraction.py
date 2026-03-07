"""
Feature Extraction Module for Partial Discharge De-noising Pipeline

This module extracts domain-specific features from PD signals to distinguish
between partial discharge events and noise. These features are used downstream
in the PD selection stage for multi-measure voting.

Five Key Measures:
1. Kurtosis: Measures signal impulsivity (peakedness)
2. Phase Consistency: Measures temporal coherence/repeatability
3. Energy Concentration: Measures frequency-domain localisation
4. Signal-to-Noise Ratio (SNR): Measures signal quality
5. Repetition Rate Regularity: Measures pattern regularity

Physics Principles:
- PD signals are impulsive (high kurtosis) Concentration of energy at peaks
- PD signals are coherent (high phase consistency) Repeatable patterns
- PD signals have localized frequency content (high energy concentration)
- PD signals have good SNR Well-defined pulse in noise
- PD events repeat (high repetition regularity) Temporal patterns
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler

# ============================================================================
# FEATURE 1: KURTOSIS - Signal Impulsivity
# ============================================================================

def compute_kurtosis(data):
    """
    Compute kurtosis of PD signals.
    
    Physics Principle:
    - Kurtosis measures the "tailedness" or peakedness of a distribution
    - Normal distribution has kurtosis = 3 (excess kurtosis = 0)
    - Impulsive signals (PD) have high kurtosis (> 3)
    - Gaussian noise has kurtosis ~ 3
    
    Interpretation:
    - PD kurtosis: typically 5-20 (sharp, concentrated peaks)
    - Noise kurtosis: typically 3-5 (distributed)
    
    Args:
        data: Input data (N, features) or (N,) for 1D
        
    Returns:
        kurtosis_values: Kurtosis for each sample
        
    Implementation Notes:
    - Excess kurtosis: kurt - 3 (shift to 0 for Gaussian)
    - Compute per-sample for clustering
    """
    
    if isinstance(data, pd.DataFrame):
        # If dataframe, compute kurtosis across features for each row
        # First, select only numeric columns to avoid TypeError with string columns
        numeric_data = data.select_dtypes(include=[np.number])
        # Average kurtosis across all numeric features
        kurtosis_vals = numeric_data.apply(
            lambda row: stats.kurtosis(row, fisher=True),  # fisher=True gives excess kurtosis
            axis=1
        )
    else:
        # Numpy array
        if len(data.shape) == 1:
            return stats.kurtosis(data, fisher=True)
        else:
            # Vectorized: compute kurtosis across features for each sample (axis=1)
            kurtosis_vals = stats.kurtosis(data, fisher=True, axis=1)
    
    return kurtosis_vals


# ============================================================================
# FEATURE 2: PHASE CONSISTENCY - Temporal Coherence
# ============================================================================

def compute_phase_consistency(data, reference_phase=None):
    """
    Compute phase consistency of signals to measure temporal coherence.
    
    Physics Principle:
    - PD events have repeatable temporal patterns  High phase consistency
    - Noise has random, uncorrelated patterns  Low phase consistency
    - Phase consistency measures how consistently signals repeat
    
    Implementation Approach:
    1. Convert signal to phase space (via Hilbert transform)
    2. Compute instantaneous phase for each sample
    3. Measure consistency across samples (via circular variance)
    
    Args:
        data: Input data (N, features) or (N,)
        reference_phase: Optional reference phase for comparison
        
    Returns:
        phase_consistency: Consistency scores for each sample (0-1)
        
    Note:
    - High consistency (0.7-1.0): PD-like (coherent patterns)
    - Low consistency (0.0-0.3): Noise-like (random)
    - Range: [0, 1] where 1 = perfect coherence
    """
    
    if isinstance(data, pd.DataFrame):
        # Select only numeric columns to avoid string columns
        numeric_data = data.select_dtypes(include=[np.number])
        data = numeric_data.values
    
    if len(data.shape) == 1:
        # Single sample - compute analytic signal
        analytic = signal.hilbert(data)
        phase = np.angle(analytic)
        # Single sample: measure phase variation
        phase_std = np.std(np.diff(phase))  # How much phase changes
        consistency = 1.0 / (1.0 + phase_std)  # Normalize to [0, 1]
        return consistency
    
    else:
        # Multiple samples - vectorized computation
        # Compute analytic signal for all samples at once (axis=1 applies per-row)
        analytic = signal.hilbert(data, axis=1)
        
        # Extract instantaneous phase for all samples
        phase = np.angle(analytic)
        
        # Vectorized phase velocity computation
        phase_velocity = np.diff(phase, axis=1)
        velocity_std = np.std(phase_velocity, axis=1)
        consistency_scores = 1.0 / (1.0 + velocity_std)
        
        return consistency_scores


# ============================================================================
# FEATURE 3: ENERGY CONCENTRATION - Frequency Localization
# ============================================================================

def compute_energy_concentration(data, freq_band=None):
    """
    Compute energy concentration in frequency domain.
    
    Physics Principle:
    - PD signals have energy concentrated in specific frequency bands
    - Typical PD frequency: 1-2 MHz (but depends on system)
    - Noise is spread across frequency spectrum
    - High energy concentration indicates PD-like signal
    
    Implementation Approach:
    1. Compute FFT of each signal
    2. Identify frequency band of interest
    3. Compute fraction of energy in that band
    4. Energy concentration = Energy_in_band / Total_energy
    
    Args:
        data: Input data (N, features) or (N,)
        freq_band: Tuple (freq_min, freq_max) for concentration band
                  Default: (0.1, 0.9) of Nyquist frequency
        
    Returns:
        energy_concentration: Fraction of energy in frequency band (0-1)
        
    Note:
    - High concentration (0.6-1.0): PD-like (localized spectrum)
    - Low concentration (0.0-0.3): Noise-like (spread spectrum)
    """
    
    if isinstance(data, pd.DataFrame):
        # Select only numeric columns to avoid string columns
        numeric_data = data.select_dtypes(include=[np.number])
        data = numeric_data.values
    
    if len(data.shape) == 1:
        # Single sample
        # Compute power spectrum
        freq, pxx = signal.periodogram(data)
        total_energy = np.sum(pxx)
        
        if freq_band is None:
            # Default: concentrate on first half of spectrum (below Nyquist)
            freq_band = (0.1 * len(freq), 0.9 * len(freq))
        
        band_mask = (freq >= freq_band[0]) & (freq <= freq_band[1])
        band_energy = np.sum(pxx[band_mask])
        
        concentration = band_energy / total_energy if total_energy > 0 else 0.0
        return concentration
    
    else:
        # Multiple samples - vectorized FFT computation
        # Compute FFT for all samples at once (axis=1 applies per-row)
        fft = np.fft.fft(data, axis=1)
        power = np.abs(fft) ** 2
        
        # Exclude upper half (by symmetry) - take first half of each row
        half_len = power.shape[1] // 2
        power = power[:, :half_len]
        
        # Total energy per sample
        total_energy = np.sum(power, axis=1)
        
        # Determine frequency band indices
        if freq_band is None:
            idx_min = int(0.1 * half_len)
            idx_max = int(0.9 * half_len)
        else:
            idx_min = int(freq_band[0] * half_len)
            idx_max = int(freq_band[1] * half_len)
        
        # Band energy for all samples
        band_energy = np.sum(power[:, idx_min:idx_max], axis=1)
        
        # Avoid division by zero
        concentrations = np.where(total_energy > 0, band_energy / total_energy, 0.0)
        
        return concentrations


# ============================================================================
# FEATURE 3B: SPECTRAL FLATNESS - Tonality vs Noise
# ============================================================================

def compute_spectral_flatness(data, eps=1e-12):
    """
    Compute spectral flatness (geometric mean / arithmetic mean of power).

    Physics Principle:
    - PD signals are peaky/band-limited -> low flatness
    - Noise is broadband -> high flatness

    Args:
        data: Input data (N, features) or (N,)
        eps: Small constant to avoid log/zero issues

    Returns:
        flatness: Spectral flatness for each sample (0-1)
    """

    if isinstance(data, pd.DataFrame):
        numeric_data = data.select_dtypes(include=[np.number])
        data = numeric_data.values

    if len(data.shape) == 1:
        fft = np.fft.fft(data)
        power = np.abs(fft[:len(data) // 2]) ** 2
        power = np.maximum(power, eps)
        geo_mean = np.exp(np.mean(np.log(power)))
        arith_mean = np.mean(power)
        return geo_mean / (arith_mean + eps)

    fft = np.fft.fft(data, axis=1)
    half_len = data.shape[1] // 2
    power = np.abs(fft[:, :half_len]) ** 2
    power = np.maximum(power, eps)
    geo_mean = np.exp(np.mean(np.log(power), axis=1))
    arith_mean = np.mean(power, axis=1)
    return geo_mean / (arith_mean + eps)


# ============================================================================
# FEATURE 3C: SPECTRAL CENTROID + BANDWIDTH - Frequency Spread
# ============================================================================

def compute_spectral_centroid_bandwidth(data, eps=1e-12):
    """
    Compute spectral centroid and bandwidth.

    Physics Principle:
    - PD signals concentrate energy in a narrower band (lower bandwidth)
    - Centroid captures dominant frequency location

    Args:
        data: Input data (N, features) or (N,)
        eps: Small constant to avoid division by zero

    Returns:
        centroid: Spectral centroid for each sample
        bandwidth: Spectral bandwidth for each sample
    """

    if isinstance(data, pd.DataFrame):
        numeric_data = data.select_dtypes(include=[np.number])
        data = numeric_data.values

    if len(data.shape) == 1:
        fft = np.fft.fft(data)
        power = np.abs(fft[:len(data) // 2]) ** 2
        freqs = np.arange(len(power))
        total = np.sum(power) + eps
        centroid = np.sum(freqs * power) / total
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total)
        return centroid, bandwidth

    fft = np.fft.fft(data, axis=1)
    half_len = data.shape[1] // 2
    power = np.abs(fft[:, :half_len]) ** 2
    freqs = np.arange(half_len)
    total = np.sum(power, axis=1) + eps
    centroid = np.sum(power * freqs, axis=1) / total
    bandwidth = np.sqrt(np.sum(((freqs - centroid[:, None]) ** 2) * power, axis=1) / total)
    return centroid, bandwidth


# ============================================================================
# FEATURE 4: SIGNAL-TO-NOISE RATIO (SNR)
# ============================================================================

def compute_snr(data, method='peak_to_rms'):
    """
    Compute Signal-to-Noise Ratio of PD signals.
    
    Physics Principle:
    - SNR = Signal Power / Noise Power
    - PD signals have high SNR (clear signal above noise floor)
    - Pure noise has SNR ~ 0
    - Typical PD SNR: 5-100 dB
    
    Implementation Approaches:
    1. Peak-to-RMS: Peak amplitude / RMS (noise floor)
    2. Spectral method: Energy in signal band / Energy in noise band
    3. Envelope method: Peak of envelope / Baseline
    
    Args:
        data: Input data (N, features) or (N,)
        method: 'peak_to_rms', 'spectral', or 'envelope'
        
    Returns:
        snr_db: SNR in decibels (dB)
        
    Note:
    - High SNR (10-50 dB): PD-like (signal clearly above noise)
    - Low SNR (0-5 dB): Noise-like (signal comparable to noise)
    """
    
    if isinstance(data, pd.DataFrame):
        # Select only numeric columns to avoid string columns
        numeric_data = data.select_dtypes(include=[np.number])
        data = numeric_data.values
    
    if len(data.shape) == 1:
        # Single sample
        if method == 'peak_to_rms':
            # Peak amplitude / RMS
            peak = np.max(np.abs(data))
            rms = np.sqrt(np.mean(data ** 2))
            snr = peak / rms if rms > 0 else 0
            snr_db = 20 * np.log10(snr + 1e-10)  # Convert to dB
            
        elif method == 'spectral':
            # Divide spectrum into signal (center) and noise (tails)
            fft = np.fft.fft(data)
            power = np.abs(fft[:len(data)//2]) ** 2
            
            # Signal: central 60% of spectrum
            signal_idx = int(0.2 * len(power)), int(0.8 * len(power))
            signal_power = np.sum(power[signal_idx[0]:signal_idx[1]])
            
            # Noise: tails (first 20% + last 20%)
            noise_power = np.sum(power[:signal_idx[0]]) + np.sum(power[signal_idx[1]:])
            
            snr = signal_power / (noise_power + 1e-10)
            snr_db = 10 * np.log10(snr + 1e-10)
            
        elif method == 'envelope':
            # Compute envelope via Hilbert transform
            analytic = signal.hilbert(data)
            envelope = np.abs(analytic)
            peak = np.max(envelope)
            baseline = np.mean(envelope)
            snr = peak / (baseline + 1e-10)
            snr_db = 20 * np.log10(snr + 1e-10)
        
        return snr_db
    
    else:
        # Multiple samples - vectorized computation
        if method == 'peak_to_rms':
            # Vectorized peak and RMS computation
            peak = np.max(np.abs(data), axis=1)
            rms = np.sqrt(np.mean(data ** 2, axis=1))
            snr = np.where(rms > 0, peak / rms, 0)
            snr_db = 20 * np.log10(snr + 1e-10)
            
        elif method == 'spectral':
            # Vectorized FFT
            fft = np.fft.fft(data, axis=1)
            half_len = data.shape[1] // 2
            power = np.abs(fft[:, :half_len]) ** 2
            
            # Signal: central 60% of spectrum
            sig_min, sig_max = int(0.2 * half_len), int(0.8 * half_len)
            signal_power = np.sum(power[:, sig_min:sig_max], axis=1)
            
            # Noise: tails
            noise_power = np.sum(power[:, :sig_min], axis=1) + np.sum(power[:, sig_max:], axis=1)
            
            snr = signal_power / (noise_power + 1e-10)
            snr_db = 10 * np.log10(snr + 1e-10)
            
        elif method == 'envelope':
            # Vectorized Hilbert transform
            analytic = signal.hilbert(data, axis=1)
            envelope = np.abs(analytic)
            peak = np.max(envelope, axis=1)
            baseline = np.mean(envelope, axis=1)
            snr = peak / (baseline + 1e-10)
            snr_db = 20 * np.log10(snr + 1e-10)
        
        return snr_db


# ============================================================================
# FEATURE 5: REPETITION RATE REGULARITY - Pattern Regularity
# ============================================================================

def compute_repetition_regularity(data, clustering_labels=None):
    """
    Compute regularity of signal repetition patterns.
    
    Physics Principle:
    - PD events often occur with regular intervals (periodic)
    - Noise is random, has no regular pattern
    - High repetition regularity indicates PD-like behavior
    
    Implementation Approach:
    1. For clustered data: Compute inter-event intervals
    2. Measure coefficient of variation (std/mean)
    3. Regularity = 1 - (CV of intervals)
    4. Range: [0, 1] where 1 = perfectly regular
    
    Args:
        data: Input data (N,) time indices or (N, features) with time info
        clustering_labels: Optional cluster assignments (N,)
        
    Returns:
        repetition_regularity: Regularity score for each cluster
        
    Note:
    - High regularity (0.7-1.0): PD-like (periodic events)
    - Low regularity (0.0-0.3): Noise-like (random)
    
    Implementation Notes:
    - Requires temporal information (sample indices or timestamps)
    - Works better with clustering information
    - Can compute at cluster level or sample level
    """
    
    if isinstance(data, pd.DataFrame):
        # Select only numeric columns to avoid string columns
        numeric_data = data.select_dtypes(include=[np.number])
        data_array = numeric_data.values
        times = np.arange(len(data))
    else:
        data_array = data
        times = np.arange(len(data))
    
    # If clustering labels provided, compute per-cluster regularity
    if clustering_labels is not None:
        regularities = {}
        unique_labels = np.unique(clustering_labels)
        
        for label in unique_labels:
            if label == -1:
                # Skip noise points
                continue
            
            # Get times for this cluster
            cluster_mask = clustering_labels == label
            cluster_times = times[cluster_mask]
            
            # Compute intervals between events
            if len(cluster_times) > 2:
                intervals = np.diff(cluster_times)
                
                # Coefficient of variation
                cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
                
                # Regularity: lower CV means more regular
                regularity = 1.0 / (1.0 + cv)
                
                regularities[label] = regularity
            else:
                regularities[label] = 0.5  # Neutral for small clusters
        
        return regularities
    
    else:
        # Single sequence: measure trend/periodicity
        # Using autocorrelation to detect periodicity
        
        # Handle 2D data: compute regularity per-sample
        if len(data_array.shape) == 2:
            # Multiple samples: compute regularity as consistency across features
            regularities = []
            for sample in data_array:
                # For each sample, compute consistency across features
                # Regularity = how similar are the features (low variance = regular)
                # Convert to float to avoid dtype issues
                sample_float = sample.astype(np.float64)
                feature_std = np.std(sample_float)
                feature_mean = np.mean(np.abs(sample_float))
                if feature_mean > 1e-10:
                    regularity = 1.0 / (1.0 + (feature_std / feature_mean))
                else:
                    regularity = 0.5
                regularities.append(regularity)
            return np.array(regularities)
        
        # 1D case: Autocorrelation
        if len(data_array) > 10:
            # Ensure data is 1D and numeric
            if len(data_array.shape) == 2:
                data_1d = data_array.flatten()
            else:
                data_1d = data_array
            
            # Convert to float to ensure compatibility
            data_1d = data_1d.astype(np.float64)
            
            autocorr = np.correlate(data_1d, data_1d, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize
            
            # Look for regular peaks in autocorrelation
            # Peak at lag τ indicates periodicity τ
            peaks, _ = signal.find_peaks(autocorr[1:], distance=5)
            
            if len(peaks) > 0:
                # High peak = high periodicity
                regularity = np.max(autocorr[peaks + 1])
            else:
                regularity = 0.0
        else:
            regularity = 0.5  # Neutral for short sequences
        
        return regularity

# ============================================================================
# FEATURE 6  COMPUTE CREST FACTOR - Peak Sharpness
# ============================================================================

def compute_crest_factor(data, verbose = True, percentile = 99.5):
    """
    Compute crest factor of PD signals.
    
    Physics Principle:
    - Crest factor = Peak amplitude / RMS
    - PD signals have high crest factor (sharp peaks)
    - Noise has low crest factor (more uniform)
    
    Args:
        data: Input data (N, features) or (N,)
        verbose: Whether to print threshold information
        percentile: Percentile for thresholding (default 99.5)
        
    Returns:
        crest_factor: Crest factor for each sample
    """ 

    if isinstance(data, pd.DataFrame):
        # Select only numeric columns to avoid string columns
        numeric_data = data.select_dtypes(include=[np.number])
        data = numeric_data.values
    if len(data.shape) == 1:
        # Single sample 
        peak = np.max(np.abs(data))
        rms = np.sqrt(np.mean(data ** 2))
        crest = peak / (rms + 1e-10)
        return crest
    else:
        # Multiple samples - vectorized computation
        peak = np.max(np.abs(data), axis=1)
        rms = np.sqrt(np.mean(data ** 2, axis=1))
        crest = peak / (rms + 1e-10)
        
        if verbose:
            threshold = np.percentile(crest, percentile)
        
        return crest

# ============================================================================
# FEATURE 7 COMPUTE FORM FACTOR 
# ============================================================================
def compute_form_factor(data, verbose = True, percentile = 99.5):
   """
   Compute form factor of PD signals.
   
   Physics Principle:
   - Form factor = RMS / Mean absolute value
   - PD signals have high form factor (sharp peaks)
   - Noise has low form factor (more uniform)

   
   Args:
       data: Input data (N, features) or (N,)
       verbose: Whether to print threshold information
       percentile: Percentile for thresholding (default 99.5)
       
   Returns:
       form_factor: Form factor for each sample
   """ 

   if isinstance(data, pd.DataFrame):
       # Select only numeric columns to avoid string columns
       numeric_data = data.select_dtypes(include=[np.number])
       data = numeric_data.values
   if len(data.shape) == 1:
       # Single sample 
       rms = np.sqrt(np.mean(data ** 2))
       mean_abs = np.mean(np.abs(data))
       form = rms / (mean_abs + 1e-10)
       return form
   else:
       # Multiple samples - vectorized computation
       rms = np.sqrt(np.mean(data ** 2, axis=1))
       mean_abs = np.mean(np.abs(data), axis=1)
       form = rms / (mean_abs + 1e-10)
       
       if verbose:
           threshold = np.percentile(form, percentile)
          
       
       return form

# ===========================================================================
# FEATURE 8 PHASE WEIGHTING - phase location of event
# ===========================================================================

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

    

# ============================================================================
# MAIN FEATURE EXTRACTION FUNCTION
# ============================================================================

def extract_pd_features(data, clustering_labels=None, raw_signals=None):
    """
    Extract all five PD-specific features from normalised data.
    
    This is the main entry point for feature extraction. Computes all
    domain-specific measures and returns them as a feature matrix.
    
    Args:
        data: Normalised PD features (N, 11)
              Expected columns: [id, energy, modfreq, area, falltime, peakwidth, 
                                phase, rise, time, peak, acq_id]
        
        clustering_labels: Optional cluster assignments (N,) for per-cluster features
        
        raw_signals: Optional raw signal data if available (for envelope analysis)
        
    Returns:
        features_df: DataFrame with shape (N, 5) containing:
                    - kurtosis: Signal impulsivity (1-20 for PD, 3-5 for noise)
                    - phase_consistency: Temporal coherence (0-1, higher for PD)
                    - energy_concentration: Frequency localization (0-1, higher for PD)
                    - snr: Signal quality in dB (10-50 for PD, 0-10 for noise)
                    - repetition_regularity: Pattern regularity (0-1, higher for PD)
    """
    
    # Pre-allocate numpy arrays for better memory efficiency and speed
    n_samples = len(data)
    features = {
        'kurtosis': np.empty(n_samples, dtype=np.float32),
        'phase_consistency': np.empty(n_samples, dtype=np.float32),
        'energy_concentration': np.empty(n_samples, dtype=np.float32),
        'snr': np.empty(n_samples, dtype=np.float32),
        'repetition_regularity': np.empty(n_samples, dtype=np.float32),
        'crest_factor': np.empty(n_samples, dtype=np.float32),
        'form_factor': np.empty(n_samples, dtype=np.float32),
        'spectral_flatness': np.empty(n_samples, dtype=np.float32),
        'spectral_centroid': np.empty(n_samples, dtype=np.float32),
        'spectral_bandwidth': np.empty(n_samples, dtype=np.float32),
        'phase_weighting': np.empty(n_samples, dtype=np.float32)
    }
    
    # ========================================================================
    # Extract each feature (results written directly to pre-allocated arrays)
    # ========================================================================
    
    print("[1/10] Computing kurtosis...")
    features['kurtosis'][:] = compute_kurtosis(data)
    
    print("[2/10] Computing phase consistency...")
    features['phase_consistency'][:] = compute_phase_consistency(data)
    
    print("[3/10] Computing energy concentration...")
    features['energy_concentration'][:] = compute_energy_concentration(data)
    
    print("[4/10] Computing SNR...")
    features['snr'][:] = compute_snr(data, method='peak_to_rms')
    
    print("[5/10] Computing repetition regularity...")
    # This one needs clustering info or time info
    if clustering_labels is not None:
        rep_reg = compute_repetition_regularity(data, clustering_labels)
        # Convert dict to array matching data order
        features['repetition_regularity'][:] = np.array([
            rep_reg.get(label, 0.5) for label in clustering_labels
        ], dtype=np.float32)
    else:
        features['repetition_regularity'][:] = compute_repetition_regularity(data)

    print("[6/10] Computing crest factor...")
    features['crest_factor'][:] = compute_crest_factor(data)

    print("[7/10] Computing form factor...")
    features['form_factor'][:] = compute_form_factor(data)

    print("[8/10] Computing spectral flatness...")
    features['spectral_flatness'][:] = compute_spectral_flatness(data)

    print("[9/10] Computing spectral centroid + bandwidth...")
    centroid, bandwidth = compute_spectral_centroid_bandwidth(data)
    features['spectral_centroid'][:] = centroid
    features['spectral_bandwidth'][:] = bandwidth

    print("[10/10] Computing phase weighting...")
    features['phase_weighting'][:] = compute_phase_weighting(data)

    # ========================================================================
    # Combine into DataFrame (zero-copy from pre-allocated arrays)
    # ========================================================================
    
    features_df = pd.DataFrame(features)
    
    print(f"Feature extraction complete: {features_df.shape}")
    
    return features_df


# ============================================================================
# FEATURE NORMALISATION & THRESHOLDING
# ============================================================================

def normalise_features(features_df):
    """
    Normalises extracted features to [0, 1] range for uniform weighting.
    
    Args:
        features_df: DataFrame from extract_pd_features()
        
    Returns:
        normalised_df: Normalised features (0-1 scale)
        scalers: Fitted scalers for inverse transformation
    """
    
    scalers = {}
    normalised_df = features_df.copy()
    
    for col in features_df.columns:
        # Min-max normalization
        min_val = features_df[col].min()
        max_val = features_df[col].max()
        
        if max_val > min_val:
            normalised_df[col] = (features_df[col] - min_val) / (max_val - min_val)
            scalers[col] = {'min': min_val, 'max': max_val}
        else:
            normalised_df[col] = 0.5  # Constant feature
            scalers[col] = {'min': min_val, 'max': min_val}
    
    return normalised_df, scalers

def get_feature_thresholds(sensor):
    """
    Return recommended thresholds for PD vs Noise classification.
    
    Thresholds are based on sensor type and principle for PD classification associated with 
    the given sensor. 

    Args: Sensor, type as string (e.g., 'HFCT', 'UHF', 'TEV')
    Sensor should be pulled from db with getSensor function 
    
    Returns:
        thresholds: Dictionary with threshold values

    """
    # TODO properly tune thresholds based on sensor characteristics
    # currently seems that the thresholds are not enhancing performance
    # no pd events are found when implementing anything other than default thresholds
    if sensor == 'HFCT':
        print(f'Using HFCT-specific feature thresholds.')
        thresholds = {
            'kurtosis': 2.5,           # Lowered from 3.0
            'phase_consistency': 0.5,  # Lowered from 0.7
            'energy_concentration': 0.4,  # Lowered from 0.6
            'snr': 3.0,                # Lowered from 5.0
            'repetition_regularity': 0.6,  # Lowered from 0.8
            'crest_factor': 5.0,         # New threshold for crest factor
            'form_factor': 1.5,          # New threshold for form factor
            'spectral_flatness': 0.3,    # New threshold for spectral flatness
            'spectral_centroid': 0.5,    # New threshold for spectral centroid
            'spectral_bandwidth': 0.4    # New threshold for spectral bandwidth
        }
    elif sensor == 'UHF':
        print(f'Using UHF-specific feature thresholds.')
        thresholds = {
            'kurtosis': 5.0,                    # Excess kurtosis > 5 suggests impulsive
            'phase_consistency': 0.65,          # Consistency > 0.65 suggests coherent
            'energy_concentration': 0.55,       # Energy > 55% in band suggests localized
            'snr': 10.0,                        # SNR > 10 dB suggests good signal quality
            'repetition_regularity': 0.8        # Regularity > 0.8 suggests periodic
        }
    elif sensor == 'TEV':
        print(f'Using TEV-specific feature thresholds.')
        thresholds = {
            'kurtosis': 3.5,                    # Excess kurtosis > 3.5 suggests impulsive
            'phase_consistency': 0.55,          # Consistency > 0.55 suggests coherent
            'energy_concentration': 0.45,       # Energy > 45% in band suggests localized
            'snr': 5.0,                         # SNR > 5 dB suggests good signal quality
            'repetition_regularity': 0.7,       # Regularity > 0.7 suggests periodic
        }
    # TODO add thresholds for HVCC 
    else:
        # default thresholds for HVCC or additional sensor types
        print(f'Using default feature thresholds.')
        thresholds = {
        'kurtosis': 3.0,                    # Excess kurtosis > 3 suggests impulsive
        'phase_consistency': 0.7,           # Consistency > 0.7 suggests coherent
        'energy_concentration': 0.6,        # Energy > 60% in band suggests localized
        'snr': 5.0,                         # SNR > 5 dB suggests good signal quality
        'repetition_regularity': 0.8        # Regularity > 0.8 suggests periodic
    }
    
    return thresholds


def get_adaptive_thresholds(cluster_stats, percentile=50):
    """
    Compute adaptive thresholds based on cluster statistics.
    "Permanent" Placeholder, until sensor specific thresholds are tuned properly 
    Perofrmance seems better with adaptive thresholds for now.
    Args:
        cluster_stats: DataFrame with aggregated cluster features
        percentile: Percentile to use for thresholding (default=50)
    """
    thresholds = {
        'kurtosis': cluster_stats['kurtosis_mean'].quantile(percentile/100),
        'phase_consistency': cluster_stats['phase_consistency_mean'].quantile(percentile/100),
        'energy_concentration': cluster_stats['energy_concentration_mean'].quantile(percentile/100),
        'snr': cluster_stats['snr_mean'].quantile(percentile/100),
        'repetition_regularity': cluster_stats['repetition_regularity_mean'].quantile(percentile/100),
        'crest_factor': cluster_stats['crest_factor_mean'].quantile(percentile/100),
        'form_factor': cluster_stats['form_factor_mean'].quantile(percentile/100),
        'spectral_flatness': cluster_stats['spectral_flatness_mean'].quantile(percentile/100),
        'spectral_centroid': cluster_stats['spectral_centroid_mean'].quantile(percentile/100),
        'spectral_bandwidth': cluster_stats['spectral_bandwidth_mean'].quantile(percentile/100),
        'phase_weighting': cluster_stats['phase_weighting_mean'].quantile(percentile/100)
    }
    return thresholds