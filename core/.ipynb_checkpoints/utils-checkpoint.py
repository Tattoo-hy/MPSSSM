# core/utils.py
"""
Utility functions for MPS-SSM experiments.
"""

import os
import torch
import numpy as np
import random
from typing import Optional

# ... (set_random_seed and EarlyStopping classes remain the same)
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0, save_path: str = None):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        if self.save_path is None:
            raise ValueError("EarlyStopping requires a 'save_path' to be provided.")
    def __call__(self, val_loss: float, model: torch.nn.Module):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            if self.best_score is not None:
                print(f"  Validation loss decreased ({abs(self.best_score):.6f} --> {abs(score):.6f}).")
            else:
                print(f"  Initial validation loss: {abs(score):.6f}.")
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    def save_checkpoint(self, model: torch.nn.Module):
        save_model(model, self.save_path)
        self.best_model_state = model.state_dict().copy()


def save_model(model: torch.nn.Module, path: str):
    """Save model to disk with its complete configuration."""
    print(f"  Saving model to disk: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # --- FIX: Save all necessary model parameters ---
    model_config = {
        'enc_in': model.enc_in,
        'pred_len': model.pred_len,
        'd_model': model.d_model,
        'n_layers': model.n_layers,
        'd_state': model.d_state,
        'expand_factor': model.expand_factor,
        'dt_rank': model.dt_rank,
        'decoder_hidden_dim': model.decoder_hidden_dim,
        'decoder_layers': model.decoder_layers,
        'lambda_val': model.lambda_val,
        'dropout': model.dropout
    }
    # -----------------------------------------------

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config
    }, path)


def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    # ... (this function remains the same)
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Model loaded from {path}")
    return model

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model: torch.nn.Module) -> float:
    param_size = 0; buffer_size = 0
    for param in model.parameters(): param_size += param.nelement() * param.element_size()
    for buffer in model.buffers(): buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024
