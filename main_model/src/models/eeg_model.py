"""
EEG-to-text model using BART's standard loss.
"""

import torch
import torch.nn as nn
from .brain_encoder import BrainRegionEncoder
from .bart_decoder import BARTDecoder


class EEGDecodingModel(nn.Module):
    def __init__(self, n_timepoints, region_channel_counts, hidden_dim=768,
                 disable_cross_region_attn=False, uniform_region_weight=False, 
                 cnn_only=False):
        super().__init__()
        
        # Brain encoder
        self.brain_encoder = BrainRegionEncoder(
            n_timepoints=n_timepoints,
            region_channel_counts=region_channel_counts,
            hidden_dim=hidden_dim,
            disable_cross_region_attn=disable_cross_region_attn,
            uniform_region_weight=uniform_region_weight,
            cnn_only=cnn_only
        )
        
        # BART decoder
        self.bart_decoder = BARTDecoder(hidden_dim=hidden_dim)

    def forward(self, eeg_data, decoder_input_ids=None, labels=None, **kwargs):
        eeg_feat = self.brain_encoder(eeg_data)
        return self.bart_decoder(
            eeg_feat=eeg_feat,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs
        )

    def generate(self, eeg_data, **kwargs):
        eeg_feat = self.brain_encoder(eeg_data)
        return self.bart_decoder.generate_from_eeg(eeg_feat, **kwargs)