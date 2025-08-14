"""
Enhanced brain region encoder for processing multi-region EEG data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv1DWithAttention


class BrainRegionEncoder(nn.Module):
    """
    Enhanced encoder for EEG data from different brain regions with multi-scale processing.
    """
    
    def __init__(self, n_timepoints, region_channel_counts, hidden_dim=768, 
                 disable_cross_region_attn=False, uniform_region_weight=False, cnn_only=False):
        super().__init__()
        self.region_names = ['frontal', 'temporal', 'central', 'parietal']
        self.region_channel_counts = region_channel_counts
        self.disable_cross_region_attn = disable_cross_region_attn
        self.uniform_region_weight = uniform_region_weight
        self.n_regions = len(self.region_names)
        self.hidden_dim = hidden_dim
        
        # Enhanced region embeddings with stronger influence
        self.region_embeddings = nn.Embedding(len(self.region_names), hidden_dim)
        nn.init.normal_(self.region_embeddings.weight, std=0.02)  # Increased std
        
        # Multi-scale temporal feature extraction
        self.temporal_scales = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]  # Different temporal scales
        ])
        
        # Feature diversity enhancement
        self.diversity_projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Enhanced region importance with dynamic weighting
        if not uniform_region_weight:
            self.region_importance = nn.Parameter(torch.randn(self.n_regions) * 0.5)
            self.region_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, self.n_regions),
                nn.Sigmoid()
            )
        
        # Per-region encoders
        self.region_encoders = nn.ModuleDict()
        for region in self.region_names:
            n_ch = region_channel_counts[region]
            self.region_encoders[region] = Conv1DWithAttention(
                n_ch, n_timepoints, hidden_dim, cnn_only=cnn_only
            )
        
        # Enhanced cross-region fusion transformer with more layers
        if not disable_cross_region_attn:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=12,  
                dim_feedforward=hidden_dim * 4,  
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2) 
            
            # Add cross-region attention mechanism
            self.cross_region_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        
        # Feature enhancement layers
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def apply_multi_scale_processing(self, x):
        """Apply multi-scale temporal processing to enhance feature diversity."""
        if len(x.shape) == 3:  # (batch, regions, hidden_dim)
            x_expanded = x.transpose(1, 2)  # (batch, hidden_dim, regions)
            
            scale_features = []
            for scale_conv in self.temporal_scales:
                scale_feat = F.gelu(scale_conv(x_expanded))
                scale_feat = scale_feat.mean(dim=2)  # (batch, hidden_dim)
                scale_features.append(scale_feat)
            
            # Combine multi-scale features
            multi_scale_feat = torch.stack(scale_features, dim=1)  # (batch, 4, hidden_dim)
            multi_scale_combined = self.diversity_projection(
                multi_scale_feat.view(multi_scale_feat.size(0), -1)
            ).unsqueeze(1)  # (batch, 1, hidden_dim)
            
            return multi_scale_combined.expand(-1, x.size(1), -1)
        else:
            return torch.zeros_like(x)

    def compute_dynamic_region_weights(self, x):
        """Compute dynamic region weights based on current features."""
        # Global feature representation
        pooled_feat = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Compute dynamic weights
        dynamic_weights = self.region_gate(pooled_feat)  # (batch, n_regions)
        
        # Combine with learned static importance
        if hasattr(self, 'region_importance'):
            static_weights = F.softmax(self.region_importance, dim=0)
            # Adaptive combination of static and dynamic weights
            alpha = 0.7  # Weight for static importance
            combined_weights = alpha * static_weights.unsqueeze(0) + (1 - alpha) * dynamic_weights
            # Renormalize
            combined_weights = F.softmax(combined_weights, dim=1)
        else:
            combined_weights = F.softmax(dynamic_weights, dim=1)
        
        return combined_weights

    def forward(self, eeg_data):
        """
        Enhanced forward pass with multi-scale processing and dynamic weighting.
        
        Args:
            eeg_data: List of 4 tensors, each of shape (batch_size, n_channels_i, n_timepoints)
        
        Returns:
            Enhanced fused feature tensor of shape (batch_size, hidden_dim)
        """
        # Extract per-region features
        feats = []
        for idx, name in enumerate(self.region_names):
            region_feat = self.region_encoders[name](eeg_data[idx])
            feats.append(region_feat)
        
        x = torch.stack(feats, dim=1)  # (batch_size, 4, hidden_dim)
        
        # Apply multi-scale processing for enhanced feature diversity
        multi_scale_features = self.apply_multi_scale_processing(x)
        x = x + 0.3 * multi_scale_features  # Weighted addition
        
        # Add region embeddings with stronger influence
        region_emb = self.region_embeddings.weight.unsqueeze(0)  # (1, 4, hidden_dim)
        x = x + 0.4 * region_emb  # Increased from 0.1
        
        # Enhanced cross-region processing
        if not self.disable_cross_region_attn:
            # Multi-layer transformer processing
            x_transformed = self.fusion_transformer(x)
            
            # Additional cross-region attention
            x_cross_attended, _ = self.cross_region_attention(
                query=x_transformed,
                key=x_transformed,
                value=x_transformed
            )
            
            # Residual connection with gating
            gate = torch.sigmoid(self.feature_enhancer(x_transformed.mean(dim=1))).unsqueeze(1)
            x = x_transformed + gate * x_cross_attended
        
        # Enhanced region weighting and fusion
        if self.uniform_region_weight or not hasattr(self, 'region_importance'):
            # Simple mean fusion
            fused = x.mean(dim=1)
        else:
            # Dynamic region weighting
            combined_weights = self.compute_dynamic_region_weights(x)  # (batch, n_regions)
            fused = (x * combined_weights.unsqueeze(-1)).sum(dim=1)
        
        # Final feature enhancement
        enhanced_fused = self.feature_enhancer(fused)
        
        # Add residual connection
        final_output = fused + 0.3 * enhanced_fused
        
        return final_output

    def get_region_weights(self):
        """Get current region importance weights with dynamic component."""
        if hasattr(self, 'region_importance') and not self.uniform_region_weight:
            static_weights = F.softmax(self.region_importance, dim=0).data.cpu().numpy()
            
            # If we have dynamic weighting, also return the static component
            return {
                'names': self.region_names,
                'softmax': static_weights,
                'has_dynamic': hasattr(self, 'region_gate')
            }
        else:
            uniform_weights = [1.0 / self.n_regions] * self.n_regions
            return {
                'names': self.region_names,
                'softmax': uniform_weights,
                'has_dynamic': False
            }
    
    def get_feature_diversity_stats(self, eeg_data):
        """Compute feature diversity statistics for monitoring."""
        with torch.no_grad():
            # Get intermediate features
            feats = []
            for idx, name in enumerate(self.region_names):
                region_feat = self.region_encoders[name](eeg_data[idx])
                feats.append(region_feat)
            
            x = torch.stack(feats, dim=1)  # (batch_size, 4, hidden_dim)
            
            # Compute pairwise similarities between regions
            x_norm = F.normalize(x, dim=-1)
            similarity_matrix = torch.matmul(x_norm, x_norm.transpose(-2, -1))
            
            # Average similarity (lower is more diverse)
            batch_avg_similarity = similarity_matrix.mean(dim=0)
            
            # Remove diagonal (self-similarity)
            mask = ~torch.eye(self.n_regions, dtype=torch.bool, device=x.device)
            diversity_score = 1.0 - batch_avg_similarity[mask].mean().item()
            
            return {
                'diversity_score': diversity_score,
                'region_similarities': batch_avg_similarity.cpu().numpy()
            }