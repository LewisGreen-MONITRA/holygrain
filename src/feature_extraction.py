"""
Feature Extraction Module for Partial Discharge De-noising Pipeline

This module extracts domain-specific features from PD signals to distinguish
between partial discharge events and noise. These features are used downstream
in the PD selection stage for multi-measure voting.

Five Key Measures:
1. Kurtosis: Measures signal impulsivity (peakedness)
2. Phase Consistency: Measures temporal coherence/repeatability
3. Energy Concentration: Measures frequency-domain localization
4. Signal-to-Noise Ratio (SNR): Measures signal quality
5. Repetition Rate Regularity: Measures pattern regularity

Physics Principles:
- PD signals are impulsive (high kurtosis) → Concentration of energy at peaks
- PD signals are coherent (high phase consistency) → Repeatable patterns
- PD signals have localized frequency content (high energy concentration)
- PD signals have good SNR → Well-defined pulse in noise
- PD events repeat (high repetition regularity) → Temporal patterns
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
    Compute kurtosis (4th moment) of PD signals.
    
    Physics Principle:
    - Kurtosis measures the "tailedness" or peakedness of a distribution
    - Normal distribution has kurtosis = 3 (excess kurtosis = 0)
    - Impulsive signals (PD) have high kurtosis (> 3)
    - Gaussian noise has kurtosis ≈ 3
    
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
            # Compute across features for each sample
            kurtosis_vals = np.array([stats.kurtosis(sample, fisher=True) for sample in data])
    
    return kurtosis_vals


# ============================================================================
# FEATURE 2: PHASE CONSISTENCY - Temporal Coherence
# ============================================================================

def compute_phase_consistency(data, reference_phase=None):
    """
    Compute phase consistency of signals to measure temporal coherence.
    
    Physics Principle:
    - PD events have repeatable temporal patterns → High phase consistency
    - Noise has random, uncorrelated patterns → Low phase consistency
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
        # Multiple samples
        consistency_scores = []
        
        for sample in data:
            # Compute analytic signal (complex representation)
            analytic = signal.hilbert(sample)
            
            # Extract instantaneous phase
            phase = np.angle(analytic)
            
            # Measure phase consistency:
            # Option 1: Circular variance (how concentrated phases are)
            # sin_mean = np.mean(np.sin(phase))
            # cos_mean = np.mean(np.cos(phase))
            # r = np.sqrt(sin_mean**2 + cos_mean**2)  # Resultant vector length
            # consistency = r  # [0, 1]
            
            # Option 2: Phase velocity regularity
            phase_velocity = np.diff(phase)
            velocity_std = np.std(phase_velocity)
            consistency = 1.0 / (1.0 + velocity_std)
            
            consistency_scores.append(consistency)
        
        return np.array(consistency_scores)


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
        # Multiple samples
        concentrations = []
        
        for sample in data:
            # Compute FFT
            fft = np.fft.fft(sample)
            power = np.abs(fft) ** 2
            
            # Exclude DC component and upper half (by symmetry)
            power = power[:len(power)//2]
            total_energy = np.sum(power)
            
            if freq_band is None:
                # Energy in central 80% of spectrum (exclude tails)
                idx_min = int(0.1 * len(power))
                idx_max = int(0.9 * len(power))
            else:
                # Convert frequency to index
                # Assume freq_band is in normalized units [0, 1]
                idx_min = int(freq_band[0] * len(power))
                idx_max = int(freq_band[1] * len(power))
            
            band_energy = np.sum(power[idx_min:idx_max])
            concentration = band_energy / total_energy if total_energy > 0 else 0.0
            
            concentrations.append(concentration)
        
        return np.array(concentrations)


# ============================================================================
# FEATURE 4: SIGNAL-TO-NOISE RATIO (SNR)
# ============================================================================

def compute_snr(data, method='peak_to_rms'):
    """
    Compute Signal-to-Noise Ratio of PD signals.
    
    Physics Principle:
    - SNR = Signal Power / Noise Power
    - PD signals have high SNR (clear signal above noise floor)
    - Pure noise has SNR ≈ 0
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
        # Multiple samples
        snr_values = []
        
        for sample in data:
            snr = compute_snr(sample, method=method)
            snr_values.append(snr)
        
        return np.array(snr_values)


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
    
    Usage:
        >>> from feature_extraction import extract_pd_features
        >>> features = extract_pd_features(normalized_data)
        >>> print(features.describe())
    """
    
    # Initialize feature dictionary
    features = {}
    
    # ========================================================================
    # Extract each feature
    # ========================================================================
    
    print("[1/5] Computing kurtosis...")
    features['kurtosis'] = compute_kurtosis(data)
    
    print("[2/5] Computing phase consistency...")
    features['phase_consistency'] = compute_phase_consistency(data)
    
    print("[3/5] Computing energy concentration...")
    features['energy_concentration'] = compute_energy_concentration(data)
    
    print("[4/5] Computing SNR...")
    features['snr'] = compute_snr(data, method='peak_to_rms')
    
    print("[5/5] Computing repetition regularity...")
    # This one needs clustering info or time info
    if clustering_labels is not None:
        rep_reg = compute_repetition_regularity(data, clustering_labels)
        # Convert dict to array matching data order
        features['repetition_regularity'] = np.array([
            rep_reg.get(label, 0.5) for label in clustering_labels
        ])
    else:
        features['repetition_regularity'] = compute_repetition_regularity(data)
    
    # ========================================================================
    # Combine into DataFrame
    # ========================================================================
    
    features_df = pd.DataFrame(features)
    
    print(f"✓ Feature extraction complete: {features_df.shape}")
    
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

def get_feature_thresholds():
    """
    Return recommended thresholds for PD vs Noise classification.
    
    These thresholds are based on domain knowledge and can be tuned
    based on your specific system characteristics.
    
    Returns:
        thresholds: Dictionary with threshold values
        
    Example:
        >>> thresholds = get_feature_thresholds()
        >>> if feature_value > thresholds['kurtosis']:
        ...     vote_for_pd = True
    """
    
    thresholds = {
        'kurtosis': 3.0,                    # Excess kurtosis > 3 suggests impulsive
        'phase_consistency': 0.7,           # Consistency > 0.7 suggests coherent
        'energy_concentration': 0.6,        # Energy > 60% in band suggests localized
        'snr': 5.0,                         # SNR > 5 dB suggests good signal quality
        'repetition_regularity': 0.8        # Regularity > 0.8 suggests periodic
    }
    
    return thresholds


# ============================================================================
# DEBUGGING & ANALYSIS
# ============================================================================

def analyze_features(features_df):
    """
    Print detailed statistics about extracted features.
    
    Useful for understanding feature distributions and tuning thresholds.
    
    Args:
        features_df: DataFrame from extract_pd_features()
    """
    
    print("\n" + "="*70)
    print("FEATURE EXTRACTION ANALYSIS")
    print("="*70 + "\n")
    
    print("Feature Statistics:")
    print("-" * 70)
    print(features_df.describe().to_string())
    
    print("\n\nFeature Correlations:")
    print("-" * 70)
    print(features_df.corr().to_string())
    
    print("\n\nSkewness & Kurtosis:")
    print("-" * 70)
    for col in features_df.columns:
        skew = stats.skew(features_df[col])
        kurt = stats.kurtosis(features_df[col])
        print(f"{col:25s}: Skew={skew:7.4f}, Kurt={kurt:7.4f}")
    
    print("\n" + "="*70)
