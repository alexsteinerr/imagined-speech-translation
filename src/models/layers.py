"""
Enhanced neural network layers for EEG processing with improved feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DWithAttention(nn.Module):
    """
    1D CNN backbone with multi-head self-attention and feature diversity mechanisms.
    """
    
    def __init__(self, n_channels, n_timepoints, hidden_dim=128, n_heads=8, cnn_only=False):
        super().__init__()
        self.cnn_only = cnn_only
        self.hidden_dim = hidden_dim
        self.n_timepoints = n_timepoints
        
        def make_residual(in_channels, out_channels):
            if in_channels == out_channels:
                return nn.Identity()
            else:
                return nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels)
                )
        
        # Enhanced CNN backbone with more sophisticated blocks
        self.conv1 = nn.Conv1d(n_channels, 128, kernel_size=9, padding=4)  # Larger kernel
        self.bn1 = nn.BatchNorm1d(128)
        self.residual1 = make_residual(n_channels, 128)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=7, padding=3)  # Larger kernel
        self.bn2 = nn.BatchNorm1d(256)
        self.residual2 = make_residual(128, 256)
        
        # Add depthwise separable convolutions for efficiency and diversity
        self.depthwise_conv = nn.Conv1d(256, 256, kernel_size=5, padding=2, groups=256)
        self.pointwise_conv = nn.Conv1d(256, 384, kernel_size=1)
        self.bn_depth = nn.BatchNorm1d(384)
        
        self.conv3 = nn.Conv1d(384, 512, kernel_size=5, padding=2)  # Larger kernel
        self.bn3 = nn.BatchNorm1d(512)
        self.residual3 = make_residual(384, 512)
        
        self.conv4 = nn.Conv1d(512, 768, kernel_size=3, padding=1)  # Increased channels
        self.bn4 = nn.BatchNorm1d(768)
        self.residual4 = make_residual(512, 768)
        
        # Add squeeze-and-excitation for channel attention
        self.se_block = SqueezeExciteBlock(768)
        
        # Enhanced dropout with different rates
        self.dropout_light = nn.Dropout(0.05)
        self.dropout_medium = nn.Dropout(0.1)
        self.dropout_heavy = nn.Dropout(0.15)
        
        if not cnn_only:
            # Enhanced CNN to attention projection
            self.cnn_to_attn = nn.Sequential(
                nn.Linear(768, hidden_dim * 2),  # Increased from 1024->768
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim, hidden_dim)  # Additional layer
            )
            
            # Enhanced learnable tokens
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            # Add multiple special tokens for richer representation
            self.temporal_tokens = nn.Parameter(torch.randn(1, 3, hidden_dim) * 0.02)
            
            # Enhanced positional embeddings
            max_seq_len = n_timepoints + 4  # +4 for special tokens
            self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
            
            # Multi-layer attention with different heads
            self.attn_layers = nn.ModuleList([
                nn.ModuleDict({
                    'attn_norm': nn.LayerNorm(hidden_dim),
                    'attn': nn.MultiheadAttention(
                        embed_dim=hidden_dim, 
                        num_heads=n_heads if i == 0 else max(4, n_heads // 2),  # Varying heads
                        dropout=0.1, 
                        batch_first=True
                    ),
                    'ffn_norm': nn.LayerNorm(hidden_dim),
                    'ffn': FeedForwardNetwork(hidden_dim, hidden_dim * (4 if i == 0 else 2))
                }) for i in range(3)  # Increased to 3 layers
            ])
            
            # Cross-attention between different temporal scales
            self.cross_scale_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads // 2,
                dropout=0.1,
                batch_first=True
            )
        
        # Enhanced projection with multiple pathways
        proj_in_dim = 768 if cnn_only else hidden_dim
        
        self.multi_scale_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.05)
            ) for _ in range(3)
        ])
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Feature diversity encouragement - FIXED: ensure dimension matches
        self.diversity_head = nn.Linear(hidden_dim, hidden_dim)  # Changed from hidden_dim // 4

    def forward(self, x):
        """
        Enhanced forward pass with multi-scale processing.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, n_timepoints)
        
        Returns:
            Enhanced feature tensor of shape (batch_size, hidden_dim)
        """
        batch_size = x.size(0)
        
        # Enhanced CNN blocks with residual connections and varying dropout
        residual = self.residual1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = x + residual
        x = F.gelu(x)
        x = self.dropout_light(x)
        
        residual = self.residual2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.gelu(x)
        x = self.dropout_light(x)
        
        # Depthwise separable convolution block
        x_depth = self.depthwise_conv(x)
        x_point = self.pointwise_conv(x_depth)
        x = self.bn_depth(x_point)
        x = F.gelu(x)
        x = self.dropout_medium(x)
        
        residual = self.residual3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + residual
        x = F.gelu(x)
        x = self.dropout_medium(x)
        
        residual = self.residual4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = x + residual
        x = F.gelu(x)
        
        # Apply squeeze-and-excitation
        x = self.se_block(x)
        x = self.dropout_heavy(x)
        
        if self.cnn_only:
            # Enhanced pooling strategy
            x = x.transpose(1, 2)  # (batch, time, channels)
            
            # Multi-scale pooling
            mean_pool = x.mean(dim=1)
            max_pool = x.max(dim=1)[0]
            
            # Attention-based pooling
            attn_weights = F.softmax(torch.sum(x * mean_pool.unsqueeze(1), dim=2), dim=1)
            attn_pool = torch.sum(x * attn_weights.unsqueeze(2), dim=1)
            
            # Multi-scale projection
            projections = []
            for i, proj_layer in enumerate(self.multi_scale_proj):
                if i == 0:
                    projections.append(proj_layer(mean_pool))
                elif i == 1:
                    projections.append(proj_layer(max_pool))
                else:
                    projections.append(proj_layer(attn_pool))
            
            combined = torch.cat(projections, dim=1)
            final_feat = self.projection(combined)
            
            # Add diversity component - FIXED: ensure proper dimensions
            diversity_feat = self.diversity_head(final_feat)
            return final_feat + 0.1 * F.normalize(diversity_feat, dim=-1)
        
        # Enhanced attention processing
        x = x.transpose(1, 2)  # (batch, time, channels)
        x = self.cnn_to_attn(x)
        
        # Add multiple special tokens
        cls = self.cls_token.expand(batch_size, -1, -1)
        temporal = self.temporal_tokens.expand(batch_size, -1, -1)
        x = torch.cat([cls, temporal, x], dim=1)
        
        # Enhanced positional embeddings
        seq_len = x.size(1)
        if seq_len <= self.pos_emb.size(1):
            x = x + self.pos_emb[:, :seq_len, :]
        else:
            # Handle sequences longer than expected
            extended_pos = self.pos_emb.repeat(1, (seq_len // self.pos_emb.size(1)) + 1, 1)
            x = x + extended_pos[:, :seq_len, :]
        
        # Multi-layer attention with cross-scale connections
        intermediate_states = []
        
        for i, layer in enumerate(self.attn_layers):
            # Self-attention
            attn_norm = layer['attn_norm'](x)
            attn_out, attn_weights = layer['attn'](attn_norm, attn_norm, attn_norm)
            x = x + self.dropout_light(attn_out)
            
            # Store intermediate state
            intermediate_states.append(x)
            
            # Feed-forward
            ffn_norm = layer['ffn_norm'](x)
            ffn_out = layer['ffn'](ffn_norm)
            x = x + self.dropout_medium(ffn_out)
            
            # Cross-scale attention (connect to previous layers)
            if i > 0 and len(intermediate_states) > 1:
                cross_attended, _ = self.cross_scale_attn(
                    query=x,
                    key=intermediate_states[-2],
                    value=intermediate_states[-2]
                )
                x = x + 0.1 * cross_attended
        
        # Enhanced final feature extraction
        cls_feat = x[:, 0, :]  # CLS token
        temporal_feat = x[:, 1:4, :].mean(dim=1)  # Average temporal tokens
        
        # Combine CLS and temporal features
        combined_feat = cls_feat + 0.3 * temporal_feat
        
        # Multi-scale projections
        projections = []
        for proj_layer in self.multi_scale_proj:
            projections.append(proj_layer(combined_feat))
        
        # Final projection
        multi_scale_combined = torch.cat(projections, dim=1)
        final_feat = self.projection(multi_scale_combined)
        
        # Add diversity component 
        diversity_feat = self.diversity_head(final_feat)
        
        return final_feat + 0.1 * F.normalize(diversity_feat, dim=-1)


class SqueezeExciteBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _ = x.size()
        
        # Squeeze
        squeezed = self.squeeze(x).view(batch_size, channels)
        
        # Excitation
        excited = self.excitation(squeezed).view(batch_size, channels, 1)
        
        # Apply attention
        return x * excited


class FeedForwardNetwork(nn.Module):
    """Enhanced feed-forward network with gating."""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Gated feed-forward
        activated = F.gelu(self.linear1(x))
        gated = torch.sigmoid(self.gate(x))
        combined = activated * gated
        combined = self.dropout(combined)
        return self.linear2(combined)