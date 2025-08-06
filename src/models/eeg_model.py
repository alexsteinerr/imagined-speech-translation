"""
Main EEG-to-Text model combining brain encoder and BART decoder.
"""

import torch
import torch.nn as nn
from .brain_encoder import BrainRegionEncoder
from .bart_decoder import BARTDecoder


class EEGDecodingModel(nn.Module):
    """
    Complete EEG-to-text model combining multi-region EEG encoding with BART decoding.
    """
    
    def __init__(self, n_timepoints, region_channel_counts, hidden_dim=768,
                 disable_cross_region_attn=False, uniform_region_weight=False, 
                 cnn_only=False, disable_cross_modal=False):
        """
        Initialize the EEG decoding model.
        
        Args:
            n_timepoints: Number of time points in EEG signals
            region_channel_counts: Dict mapping region names to channel counts
            hidden_dim: Hidden dimension size
            disable_cross_region_attn: Whether to disable cross-region attention
            uniform_region_weight: Whether to use uniform region weighting
            cnn_only: Whether to use CNN-only processing (no attention)
            disable_cross_modal: Whether to disable cross-modal connections
        """
        super().__init__()
        
        # Brain encoder for EEG processing
        self.brain_encoder = BrainRegionEncoder(
            n_timepoints=n_timepoints,
            region_channel_counts=region_channel_counts,
            hidden_dim=hidden_dim,
            disable_cross_region_attn=disable_cross_region_attn,
            uniform_region_weight=uniform_region_weight,
            cnn_only=cnn_only
        )
        
        # BART decoder for text generation
        self.bart_decoder = BARTDecoder(
            hidden_dim=hidden_dim, 
            disable_cross_modal=disable_cross_modal
        )
        
        # Training step counter
        self.register_buffer('training_step', torch.tensor(0))

    def forward(self, eeg_data, decoder_input_ids=None, labels=None, **kwargs):
        """
        Forward pass through the model.
        
        Args:
            eeg_data: List of 4 EEG tensors for different brain regions
            decoder_input_ids: Input token IDs for decoder
            labels: Target token IDs for training loss
            **kwargs: Additional arguments
        
        Returns:
            Model outputs including loss and logits
        """
        # Encode EEG data
        eeg_feat = self.brain_encoder(eeg_data)
        
        # Increment training step counter
        if self.training:
            self.training_step += 1
            
        # Decode to text
        return self.bart_decoder(
            eeg_feat=eeg_feat,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs
        )

    def generate(self, eeg_data, **kwargs):
        """
        Generate text from EEG data.
        
        Args:
            eeg_data: List of 4 EEG tensors for different brain regions
            **kwargs: Generation parameters (max_length, num_beams, etc.)
        
        Returns:
            Generated token IDs
        """
        eeg_feat = self.brain_encoder(eeg_data)
        return self.bart_decoder.generate_from_eeg(eeg_feat, **kwargs)

    def get_region_weights(self):
        """Get current region importance weights."""
        return self.brain_encoder.get_region_weights()
    
    def get_model_info(self):
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'region_weights': self.get_region_weights(),
            'training_step': self.training_step.item()
        }