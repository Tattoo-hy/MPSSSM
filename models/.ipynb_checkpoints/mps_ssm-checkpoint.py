# models/mps_ssm.py
"""
MPS-SSM (Minimal Predictive Sufficiency State Space Model) Implementation
Modified for multivariate time series prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class SelectiveSSMBlock(nn.Module):
    """
    Core SSM block with selective mechanism based on Mamba architecture
    Enhanced with MPS principles
    """
    
    def __init__(self, 
                 d_model: int,
                 d_state: int = 16,
                 expand_factor: int = 2,
                 dt_rank: str = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank
            
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ... (This part is correct and remains unchanged)
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = F.silu(x)
        x_proj = self.x_proj(x)
        delta_proj, B, C = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_proj))
        A = -torch.exp(self.A_log.float())
        hidden_states = self._ssm_recurrence(x, A, B, C, delta)
        y = hidden_states + x * self.D
        y = y * F.silu(z)
        output = self.out_proj(y)
        return output, hidden_states
        
    def _ssm_recurrence(self, x, A, B, C, delta):
        batch, seq_len, d_inner = x.shape
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        h = torch.zeros(batch, d_inner, A.shape[1], device=x.device)
        outputs = []
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y)
        return torch.stack(outputs, dim=1)


class MinimalityRegularizer(nn.Module):
    # ... (This class is correct and remains unchanged)
    def __init__(self, hidden_dim: int, input_dim: int, decoder_hidden_dim: int = 256, decoder_layers: int = 2):
        super().__init__()
        layers = []
        current_dim = hidden_dim
        for i in range(decoder_layers):
            if i == decoder_layers - 1:
                layers.append(nn.Linear(current_dim, input_dim))
            else:
                layers.append(nn.Linear(current_dim, decoder_hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                current_dim = decoder_hidden_dim
        self.decoder = nn.Sequential(*layers)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden_states)


class MPSSSM(nn.Module):
    """
    MPS-SSM: Minimal Predictive Sufficiency State Space Model
    """
    
    def __init__(self,
                 enc_in: int,
                 pred_len: int,
                 d_model: int = 512,
                 n_layers: int = 4,
                 d_state: int = 16,
                 expand_factor: int = 2,
                 dt_rank: str = "auto",
                 decoder_hidden_dim: int = 256,
                 decoder_layers: int = 2,
                 lambda_val: float = 0.01,
                 dropout: float = 0.1):
        super().__init__()
        
        # --- FIX: Store all configuration parameters as instance attributes ---
        self.enc_in = enc_in
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.dt_rank = dt_rank
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_layers = decoder_layers
        self.lambda_val = lambda_val
        self.dropout = dropout
        # --------------------------------------------------------------------
        
        self.input_embed = nn.Linear(enc_in, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        self.ssm_blocks = nn.ModuleList([
            SelectiveSSMBlock(d_model=d_model, d_state=d_state, expand_factor=expand_factor, dt_rank=dt_rank)
            for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        
        self.minimality_regularizer = MinimalityRegularizer(
            hidden_dim=d_model, input_dim=enc_in,
            decoder_hidden_dim=decoder_hidden_dim, decoder_layers=decoder_layers
        )
        
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len * enc_in)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # ... (This part is correct and remains unchanged)
        batch_size, seq_len, _ = x.shape
        x_embed = self.embed_dropout(self.input_embed(x))
        original_input = x.clone()
        hidden = x_embed
        all_hidden_states = []
        for i, (ssm_block, layer_norm) in enumerate(zip(self.ssm_blocks, self.layer_norms)):
            ssm_out, ssm_hidden = ssm_block(hidden)
            hidden = layer_norm(hidden + ssm_out)
            all_hidden_states.append(hidden)
        final_hidden = hidden[:, -1, :]
        predictions = self.pred_head(final_hidden).view(batch_size, self.pred_len, self.enc_in)
        
        middle_layer_idx = len(self.ssm_blocks) // 2
        reconstructed_input = self.minimality_regularizer(all_hidden_states[middle_layer_idx])
        
        return {'prediction': predictions, 'reconstructed_input': reconstructed_input, 'original_input_for_decoder': original_input, 'hidden_states': hidden}
        
    def compute_loss(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # ... (This part is correct and remains unchanged)
        pred_loss = F.mse_loss(outputs['prediction'], target)
        min_loss = F.mse_loss(outputs['reconstructed_input'], outputs['original_input_for_decoder'])
        total_loss = pred_loss + self.lambda_val * min_loss
        return {'total_loss': total_loss, 'pred_loss': pred_loss, 'min_loss': min_loss}
