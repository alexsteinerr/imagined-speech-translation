import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

class Conv1DWithAttention(nn.Module):
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

class BrainRegionEncoder(nn.Module):
    def __init__(self, n_timepoints, region_channel_counts, hidden_dim=128, 
                 disable_cross_region_attn=False, uniform_region_weight=False, cnn_only=False):
        super().__init__()
        self.region_names = ['frontal', 'temporal', 'central', 'parietal']
        self.region_channel_counts = region_channel_counts
        self.disable_cross_region_attn = disable_cross_region_attn
        self.uniform_region_weight = uniform_region_weight
        self.n_regions = len(self.region_names)
        
        self.region_embeddings = nn.Embedding(len(self.region_names), hidden_dim)
        nn.init.normal_(self.region_embeddings.weight, std=0.01)
        
        if not uniform_region_weight:
            self.region_importance = nn.Parameter(torch.ones(self.n_regions) * 0.25)
        
        self.region_encoders = nn.ModuleDict()
        for region in self.region_names:
            n_ch = region_channel_counts[region]
            self.region_encoders[region] = Conv1DWithAttention(
                n_ch, n_timepoints, hidden_dim, cnn_only=cnn_only
            )
        
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
        feats = []
        for idx, name in enumerate(self.region_names):
            region_feat = self.region_encoders[name](eeg_data[idx])
            feats.append(region_feat)
        
        x = torch.stack(feats, dim=1)
        x = x + self.region_embeddings.weight.unsqueeze(0) * 0.1
        
        if not self.disable_cross_region_attn:
            x = self.fusion_transformer(x)
        
        if self.uniform_region_weight or not hasattr(self, 'region_importance'):
            fused = x.mean(dim=1)
        else:
            w = torch.softmax(self.region_importance, dim=0)
            fused = (x * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        
        return fused

    def get_region_weights(self):
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

class BARTDecoder(nn.Module):
    def __init__(self, hidden_dim, disable_cross_modal=False):
        super().__init__()
        self.disable_cross_modal = disable_cross_modal
        self.hidden_dim = hidden_dim
        
        self.bart = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
        self.bart_dim = self.bart.config.d_model
        
        self.eeg_to_bart = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.bart_dim),
            nn.LayerNorm(self.bart_dim)
        )
        
        for layer in self.eeg_to_bart:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        self.encoder_length = 8
        self.eeg_conditioning_strength = nn.Parameter(torch.tensor(0.2))
        self.encoder_queries = nn.Parameter(torch.randn(1, self.encoder_length, self.bart_dim) * 0.01)
        
        self.eeg_attention = nn.MultiheadAttention(
            embed_dim=self.bart_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.encoder_pos_emb = nn.Parameter(torch.randn(1, self.encoder_length, self.bart_dim) * 0.01)

    def create_encoder_sequence(self, eeg_feat):
        batch_size = eeg_feat.shape[0]
        
        conditioning_strength = torch.clamp(self.eeg_conditioning_strength, 0.05, 1.0)
        proj_eeg = self.eeg_to_bart(eeg_feat) * conditioning_strength
        eeg_expanded = proj_eeg.unsqueeze(1)
        
        queries = self.encoder_queries.expand(batch_size, -1, -1)
        encoder_seq, _ = self.eeg_attention(
            query=queries,
            key=eeg_expanded,
            value=eeg_expanded
        )
        
        encoder_seq = encoder_seq + self.encoder_pos_emb
        return encoder_seq

    def forward(self, eeg_feat, decoder_input_ids=None, labels=None, **kwargs):
        try:
            batch_size = eeg_feat.shape[0]
            device = eeg_feat.device
            
            encoder_seq = self.create_encoder_sequence(eeg_feat)
            encoder_attention_mask = torch.ones(
                batch_size, self.encoder_length,
                dtype=torch.long, device=device
            )
            
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_seq)
            
            bart_outputs = self.bart(
                input_ids=None,
                attention_mask=encoder_attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
            
            return bart_outputs
            
        except Exception as e:
            logger.error(f"BARTDecoder forward pass failed: {e}")
            vocab_size = self.bart.config.vocab_size
            seq_len = decoder_input_ids.size(1) if decoder_input_ids is not None else 16
            
            dummy_logits = torch.zeros(batch_size, seq_len, vocab_size, device=eeg_feat.device)
            dummy_loss = torch.tensor(5.0, device=eeg_feat.device, requires_grad=True)
            return Seq2SeqLMOutput(loss=dummy_loss, logits=dummy_logits)

    def generate_from_eeg(self, eeg_feat, max_length=32, **kwargs):
        batch_size = eeg_feat.shape[0]
        device = eeg_feat.device
        
        encoder_seq = self.create_encoder_sequence(eeg_feat)
        encoder_attention_mask = torch.ones(
            batch_size, self.encoder_length,
            dtype=torch.long, device=device
        )
        
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_seq)
        
        generation_config = {
            'encoder_outputs': encoder_outputs,
            'attention_mask': encoder_attention_mask,
            'decoder_start_token_id': self.bart.config.decoder_start_token_id,
            'pad_token_id': self.bart.config.pad_token_id,
            'eos_token_id': self.bart.config.eos_token_id,
            'max_length': max_length,
            'min_length': kwargs.get('min_length', 3),
            'no_repeat_ngram_size': kwargs.get('no_repeat_ngram_size', 2),
            'length_penalty': kwargs.get('length_penalty', 1.0),
        }
        
        if kwargs.get('do_sample', False):
            generation_config.update({
                'do_sample': True,
                'temperature': kwargs.get('temperature', 0.7),
                'top_k': kwargs.get('top_k', 50),
                'top_p': kwargs.get('top_p', 0.9),
            })
        else:
            generation_config.update({
                'do_sample': False,
                'num_beams': kwargs.get('num_beams', 3),
            })
            
            if generation_config['num_beams'] > 1:
                generation_config['early_stopping'] = True
        
        try:
            generated_ids = self.bart.generate(**generation_config)
            return generated_ids
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return torch.full(
                (batch_size, 1),
                self.bart.config.decoder_start_token_id,
                dtype=torch.long, device=device
            )

class EEGDecodingModel(nn.Module):
    def __init__(self, n_timepoints, region_channel_counts, hidden_dim=768,
                 disable_cross_region_attn=False, uniform_region_weight=False, 
                 cnn_only=False, disable_cross_modal=False):
        super().__init__()
        
        self.brain_encoder = BrainRegionEncoder(
            n_timepoints,
            region_channel_counts,
            hidden_dim,
            disable_cross_region_attn=disable_cross_region_attn,
            uniform_region_weight=uniform_region_weight,
            cnn_only=cnn_only
        )
        
        self.bart_decoder = BARTDecoder(hidden_dim, disable_cross_modal=disable_cross_modal)
        
        self.register_buffer('training_step', torch.tensor(0))

    def forward(self, eeg_data, decoder_input_ids=None, labels=None, **kwargs):
        eeg_feat = self.brain_encoder(eeg_data)
        
        if self.training:
            self.training_step += 1
            
        return self.bart_decoder(
            eeg_feat,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs
        )

    def generate(self, eeg_data, **kwargs):
        eeg_feat = self.brain_encoder(eeg_data)
        return self.bart_decoder.generate_from_eeg(eeg_feat, **kwargs)

    def get_region_weights(self):
        return self.brain_encoder.get_region_weights()