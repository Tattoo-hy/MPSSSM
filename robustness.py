"""
Robustness testing utilities for MPS-SSM
Implements various noise injection methods for evaluating model robustness
"""

import numpy as np
from typing import Optional


def add_impulse_noise(data: np.ndarray, 
                     scale: np.ndarray,
                     probability: float = 0.05,
                     magnitude_factor: float = 5.0) -> np.ndarray:
    """
    Add impulse noise to time series data
    
    Args:
        data: Input data of shape (seq_len, n_features)
        scale: Standard deviation of each feature (from scaler)
        probability: Probability of impulse at each time point
        magnitude_factor: Impulse magnitude as multiple of std dev
        
    Returns:
        Noisy data with same shape as input
    """
    noisy_data = data.copy()
    seq_len, n_features = data.shape
    
    # Generate random impulse locations
    impulse_mask = np.random.random((seq_len, n_features)) < probability
    
    # Generate impulse magnitudes (both positive and negative)
    impulse_signs = np.random.choice([-1, 1], size=(seq_len, n_features))
    impulse_values = impulse_signs * magnitude_factor * scale[np.newaxis, :]
    
    # Apply impulses
    noisy_data[impulse_mask] += impulse_values[impulse_mask]
    
    return noisy_data


def add_spurious_correlation(data: np.ndarray,
                           scale: np.ndarray,
                           frequency: float = 0.1,
                           amplitude_factor: float = 0.5,
                           correlation_strength: float = 0.7) -> np.ndarray:
    """
    Add spurious correlation that appears in history but not in future
    This simulates non-causal disturbances that correlate with past data
    
    Args:
        data: Input data of shape (seq_len, n_features)
        scale: Standard deviation of each feature
        frequency: Frequency of spurious signal (relative to sequence length)
        amplitude_factor: Amplitude relative to signal std
        correlation_strength: How strongly the spurious signal correlates with history
        
    Returns:
        Noisy data with spurious correlations added
    """
    noisy_data = data.copy()
    seq_len, n_features = data.shape
    
    # Generate base spurious signal (high-frequency sinusoid)
    t = np.arange(seq_len)
    base_signal = np.sin(2 * np.pi * frequency * t)
    
    # Make the spurious signal correlate with historical patterns
    # by modulating it with a smoothed version of the data
    for feat_idx in range(n_features):
        # Smooth the feature to extract low-frequency components
        window_size = max(5, seq_len // 20)
        smoothed = np.convolve(data[:, feat_idx], 
                              np.ones(window_size) / window_size, 
                              mode='same')
        
        # Create spurious signal that correlates with smoothed history
        spurious = base_signal * amplitude_factor * scale[feat_idx]
        
        # Modulate spurious signal to correlate with historical patterns
        # This creates a signal that "looks" related to the data but isn't causal
        correlation_factor = correlation_strength * (smoothed - smoothed.mean()) / (smoothed.std() + 1e-8)
        modulated_spurious = spurious * (1 + correlation_factor)
        
        # Add to data
        noisy_data[:, feat_idx] += modulated_spurious
        
    return noisy_data


def add_gaussian_noise(data: np.ndarray,
                      scale: np.ndarray,
                      noise_level: float = 0.1) -> np.ndarray:
    """
    Add Gaussian noise to data (for baseline comparison)
    
    Args:
        data: Input data
        scale: Standard deviation of each feature
        noise_level: Noise standard deviation as fraction of signal std
        
    Returns:
        Noisy data
    """
    noise = np.random.randn(*data.shape) * (noise_level * scale[np.newaxis, :])
    return data + noise


def add_structured_missing(data: np.ndarray,
                         missing_rate: float = 0.1,
                         burst_length: int = 5) -> np.ndarray:
    """
    Add structured missing values (consecutive missing points)
    
    Args:
        data: Input data
        missing_rate: Overall fraction of missing data
        burst_length: Average length of missing bursts
        
    Returns:
        Data with NaN values for missing points
    """
    noisy_data = data.copy()
    seq_len = data.shape[0]
    
    # Calculate number of bursts
    n_missing = int(seq_len * missing_rate)
    n_bursts = max(1, n_missing // burst_length)
    
    for _ in range(n_bursts):
        # Random burst start and length
        start = np.random.randint(0, seq_len - burst_length)
        length = np.random.randint(1, min(burst_length * 2, seq_len - start))
        
        # Set to NaN
        noisy_data[start:start + length] = np.nan
        
    return noisy_data
