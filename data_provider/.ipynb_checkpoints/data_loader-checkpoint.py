# data_provider/data_loader.py
"""
Data loading utilities for time series datasets
Supports ETT, Weather, and Traffic datasets with multivariate prediction
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Optional, Callable, Dict, Any
from pathlib import Path


class TimeSeriesDataset(Dataset):
    """
    General time series dataset for multivariate forecasting
    """
    
    def __init__(self, 
                 data_path: str,
                 mode: str = 'train',
                 seq_len: int = 96,
                 pred_len: int = 96,
                 dataset_type: str = 'ETT',
                 noise_fn: Optional[Callable] = None):
        self.data_path = data_path
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dataset_type = dataset_type
        self.noise_fn = noise_fn
        
        self._read_data()
        
    def _read_data(self):
        """Read and preprocess data based on dataset type"""
        df = pd.read_csv(self.data_path)
        
        if self.dataset_type.startswith('ETT'):
            data = df.iloc[:, 1:].values
        elif self.dataset_type == 'weather':
            data = df.iloc[:, 1:].values
        elif self.dataset_type == 'traffic':
            data = df.iloc[:, 1:].values
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        self.n_features = data.shape[1]
        
        if self.dataset_type == 'traffic':
            train_len, val_len = int(0.7 * len(data)), int(0.2 * len(data))
        else:
            train_len, val_len = int(0.7 * len(data)), int(0.1 * len(data))
        
        self.scaler = StandardScaler()
        scaler_path = Path(self.data_path.replace('.csv', '_scaler.npz'))
        lock_path = Path(str(scaler_path) + '.lock')

        if not scaler_path.exists():
            self._create_scaler_with_lock(data[:train_len], scaler_path, lock_path)
        
        self._load_scaler_params(scaler_path)

        if self.mode == 'train':
            self.data = data[:train_len]
        elif self.mode == 'val':
            self.data = data[train_len : train_len + val_len]
        else:  # test
            self.data = data[train_len + val_len:]

        self.data = self.scaler.transform(self.data)

    def _create_scaler_with_lock(self, train_data: np.ndarray, scaler_path: Path, lock_path: Path):
        """Atomically create the scaler file using a file lock."""
        while True:
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    if not scaler_path.exists():
                        self.scaler.fit(train_data)
                        np.savez(scaler_path, mean=self.scaler.mean_, scale=self.scaler.scale_)
                finally:
                    os.close(lock_fd)
                    os.unlink(lock_path)
                break
            except FileExistsError:
                time.sleep(random.uniform(0.5, 1.5))
            except Exception:
                if 'lock_fd' in locals() and lock_fd:
                    os.close(lock_fd)
                    os.unlink(lock_path)
                raise

    def _load_scaler_params(self, scaler_path: Path):
        """Load scaler parameters from the specified path."""
        if scaler_path.exists():
            params = np.load(scaler_path)
            self.scaler.mean_ = params['mean']
            self.scaler.scale_ = params['scale']
        else:
            raise FileNotFoundError(f"Scaler file not found at {scaler_path} after lock mechanism.")
            
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_len].copy()
        
        if self.noise_fn is not None:
            input_seq = self.noise_fn(input_seq)
            
        target_seq = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        
        return (torch.FloatTensor(input_seq), torch.FloatTensor(target_seq))


def get_dataloader(config: Dict[str, Any], mode: str = 'train') -> DataLoader:
    """Create dataloader for specified mode"""
    dataset_name = config['dataset']
    if dataset_name.startswith('ETT'):
        dataset_type = 'ETT'
    elif dataset_name == 'weather':
        dataset_type = 'weather'
    elif dataset_name == 'traffic':
        dataset_type = 'traffic'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    data_file = os.path.join(config['data_path'], f"{dataset_name}.csv")
    
    dataset = TimeSeriesDataset(
        data_path=data_file, mode=mode,
        seq_len=config['seq_len'], pred_len=config['pred_len'],
        dataset_type=dataset_type
    )
    
    # --- FIX: Set num_workers to 0 to prevent hanging child processes ---
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=(mode == 'train'),
        num_workers=8, # This is the critical change
        drop_last=False
    )
    # --------------------------------------------------------------------
    
    return dataloader

# For backward compatibility
ETTDataset = TimeSeriesDataset
