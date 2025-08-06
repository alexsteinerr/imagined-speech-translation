"""
Neural network layers for EEG processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DWithAttention(nn.Module):
    """
    1D CNN backbone with optional self-attention for EEG signal processing.
    """
    
    def __init__(self, n_channels, n_timepoints, hidden_dim=128, n_heads=8, cnn_only=False):
        super().__init__()
        self.cnn_only = cnn_only
        
        def make_residual(in_channels, out_channels):
            if in_channels == out_channels:
                return nn.Identity()
            else:
                return nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels)
                )
        
        # CNN backbone
        self.conv1 = nn.Conv1d(n_channels, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.residual1 = make_residual(n_channels, 128)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.residual2 = make_residual(128, 256)
        
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.residual3 = make_residual(256, 512)
        
        self.conv4 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(1024)
        self.residual4 = make_residual(512, 1024)
        
        self.dropout = nn.Dropout(0.1)
        
        if not cnn_only:
            self.cnn_to_attn = nn.Sequential(
                nn.Linear(1024, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(768, 512)
            )
            
            self.cls_token = nn.Parameter(torch.randn(1, 1, 512) * 0.01)
            self.pos_emb = nn.Parameter(torch.randn(1, n_timepoints + 1, 512) * 0.01)
            
            self.attn_layers = nn.ModuleList([
                nn.ModuleDict({
                    'attn_norm': nn.LayerNorm(512),
                    'attn': nn.MultiheadAttention(embed_dim=512, num_heads=n_heads, dropout=0.1, batch_first=True),
                    'ffn_norm': nn.LayerNorm(512),
                    'ffn': nn.Sequential(
                        nn.Linear(512, 1024),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(1024, 512)
                    )
                }) for _ in range(2)
            ])
        
        proj_in_dim = 1024 if cnn_only else 512
        self.projection = nn.Sequential(
            nn.Linear(proj_in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, n_channels, n_timepoints)
        
        Returns:
            Feature tensor of shape (batch_size, hidden_dim)
        """
        # CNN blocks with residual connections
        residual = self.residual1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = x + residual
        x = F.gelu(x)
        x = self.dropout(x)
        
        residual = self.residual2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.gelu(x)
        x = self.dropout(x)
        
        residual = self.residual3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + residual
        x = F.gelu(x)
        x = self.dropout(x)
        
        residual = self.residual4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = x + residual
        x = F.gelu(x)
        x = self.dropout(x)
        
        if self.cnn_only:
            x = x.transpose(1, 2)
            mean_pool = x.mean(dim=1)
            max_pool = x.max(dim=1)[0]
            pooled = (mean_pool + max_pool) / 2
            return self.projection(pooled)
        
        # Attention processing
        x = x.transpose(1, 2)
        x = self.cnn_to_attn(x)
        
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb
        
        for layer in self.attn_layers:
            attn_norm = layer['attn_norm'](x)
            attn_out, _ = layer['attn'](attn_norm, attn_norm, attn_norm)
            x = x + self.dropout(attn_out)
            
            ffn_norm = layer['ffn_norm'](x)
            ffn_out = layer['ffn'](ffn_norm)
            x = x + self.dropout(ffn_out)
        
        cls_feat = x[:, 0, :]
        return self.projection(cls_feat)