# core/metrics.py
"""
Evaluation metrics for MPS-SSM
"""

import numpy as np
import torch
from typing import Union


def calculate_metrics(predictions: Union[np.ndarray, torch.Tensor],
                     targets: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    Calculate various evaluation metrics
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        Dictionary containing metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (with small epsilon to avoid division by zero)
    epsilon = 1e-8
    mape = np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100
    
    # Calculate correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation
    }
