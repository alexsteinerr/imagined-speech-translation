"""
Brain region encoder for processing multi-region EEG data.
"""

import torch
import torch.nn as nn
from .layers import Conv1DWithAttention


class BrainRegionEncoder(nn.Module):
    """
    Encodes EEG data from different brain regions and fuses them.
    """
    
    def __init__(self, n_timepoints, region_channel_counts, hidden_dim=128, 
                 disable_cross_region_attn=False, uniform_region_weight=False, cnn_only=False):
        super().__init__()
        self.region_names = ['frontal', 'temporal', 'central', 'parietal']
        self.region_channel_counts = region_channel_counts
        self.disable_cross_region_attn = disable_cross_region_attn
        self.uniform_region_weight = uniform_region_weight
        self.n_regions = len(self.region_names)
        
        # Region embeddings
        self.region_embeddings = nn.Embedding(len(self.region_names), hidden_dim)
        nn.init.normal_(self.region_embeddings.weight, std=0.01)
        
        # Learnable region importance weights
        if not uniform_region_weight:
            self.region_importance = nn.Parameter(torch.ones(self.n_regions) * 0.25)
        
        # Per-region encoders
        self.region_encoders = nn.ModuleDict()
        for region in self.region_names:
            n_ch = region_channel_counts[region]
            self.region_encoders[region] = Conv1DWithAttention(
                n_ch, n_timepoints, hidden_dim, cnn_only=cnn_only
            )
        
        # Cross-region fusion transformer
        if not disable_cross_region_attn:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.fusion_transformer = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, eeg_data):
        """
        Args:
            eeg_data: List of 4 tensors, each of shape (batch_size, n_channels_i, n_timepoints)
        
        Returns:
            Fused feature tensor of shape (batch_size, hidden_dim)
        """
        feats = []
        for idx, name in enumerate(self.region_names):
            region_feat = self.region_encoders[name](eeg_data[idx])
            feats.append(region_feat)
        
        x = torch.stack(feats, dim=1)  # (batch_size, 4, hidden_dim)
        
        # Add region embeddings
        x = x + self.region_embeddings.weight.unsqueeze(0) * 0.1
        
        # Cross-region attention fusion
        if not self.disable_cross_region_attn:
            x = self.fusion_transformer(x)
        
        # Region weighting and fusion
        if self.uniform_region_weight or not hasattr(self, 'region_importance'):
            fused = x.mean(dim=1)
        else:
            w = torch.softmax(self.region_importance, dim=0)
            fused = (x * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        
        return fused

    def get_region_weights(self):
        """Get current region importance weights."""
        if hasattr(self, 'region_importance') and not self.uniform_region_weight:
            raw_weights = self.region_importance.data.cpu().numpy()
            softmax_weights = torch.softmax(self.region_importance, dim=0).data.cpu().numpy()
            return {
                'names': self.region_names,
                'raw': raw_weights,
                'softmax': softmax_weights
            }
        else:
            uniform_weights = [1.0 / self.n_regions] * self.n_regions
            return {
                'names': self.region_names,
                'raw': uniform_weights,
                'softmax': uniform_weights
            }