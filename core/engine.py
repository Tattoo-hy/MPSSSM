# core/engine.py
"""
Training and evaluation engine for MPS-SSM
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple, Optional


def train_one_epoch(model: nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   device: torch.device,
                   lambda_val: float) -> float:
    """
    Train model for one epoch
    
    Args:
        model: MPS-SSM model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to use
        lambda_val: Lambda value for minimality regularization
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_min_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute losses
        losses = model.compute_loss(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics
        total_loss += losses['total_loss'].item()
        total_pred_loss += losses['pred_loss'].item()
        total_min_loss += losses['min_loss'].item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'pred': f"{losses['pred_loss'].item():.4f}",
            'min': f"{losses['min_loss'].item():.4f}"
        })
        
    avg_loss = total_loss / num_batches
    print(f"  Avg train loss: {avg_loss:.4f} "
          f"(pred: {total_pred_loss/num_batches:.4f}, "
          f"min: {total_min_loss/num_batches:.4f})")
    
    return avg_loss


def evaluate(model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on validation/test set
    
    Args:
        model: MPS-SSM model
        dataloader: Evaluation dataloader
        device: Device to use
        
    Returns:
        (MSE, MAE) tuple
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            predictions = outputs['prediction']
            
            # Calculate metrics
            mse = torch.nn.functional.mse_loss(predictions, targets, reduction='sum')
            mae = torch.nn.functional.l1_loss(predictions, targets, reduction='sum')
            
            total_mse += mse.item()
            total_mae += mae.item()
            total_samples += targets.numel()
            
    # Average over all samples
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    return avg_mse, avg_mae
