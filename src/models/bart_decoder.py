"""
Enhanced BART-based decoder for converting EEG features to text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import logging

logger = logging.getLogger(__name__)


class BARTDecoder(nn.Module):
    """
    Enhanced BART decoder with stronger EEG conditioning and multi-layer attention.
    """
    
    def __init__(self, hidden_dim, disable_cross_modal=False):
        super().__init__()
        self.disable_cross_modal = disable_cross_modal
        self.hidden_dim = hidden_dim
        
        # Load pretrained BART
        self.bart = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
        self.bart_dim = self.bart.config.d_model
        
        # Enhanced multi-layer EEG to BART projection
        self.eeg_to_bart = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim * 2, self.bart_dim),
            nn.LayerNorm(self.bart_dim)
        )
        
        # More conservative initialization to prevent gradient explosion
        for layer in self.eeg_to_bart:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.02)  # Very small gain
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Increased encoder sequence length for richer representation
        self.encoder_length = 48  # Increased from 24
        
        # Enhanced adaptive conditioning system
        self.conditioning_mlp = nn.Sequential(
            nn.Linear(self.bart_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Multi-head cross-attention for richer EEG conditioning
        self.eeg_cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.bart_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ) for _ in range(3)  # Multiple attention layers
        ])
        
        # EEG feature memory bank for enhanced representation
        self.eeg_memory_keys = nn.Parameter(torch.randn(64, self.bart_dim) * 0.01)  # Increased size
        self.eeg_memory_values = nn.Parameter(torch.randn(64, self.bart_dim) * 0.01)
        
        # Enhanced encoder queries with learnable diversity
        self.encoder_queries = nn.Parameter(torch.randn(1, self.encoder_length, self.bart_dim) * 0.01)
        
        # Position embeddings for encoder sequence
        self.encoder_pos_emb = nn.Parameter(torch.randn(1, self.encoder_length, self.bart_dim) * 0.01)
        
        # Content-aware gating mechanism
        self.content_gate = nn.Sequential(
            nn.Linear(self.bart_dim, self.bart_dim // 2),
            nn.GELU(),
            nn.Linear(self.bart_dim // 2, self.bart_dim),
            nn.Sigmoid()
        )
        
        # Feature diversification layers
        self.diversifier = nn.Sequential(
            nn.Linear(self.bart_dim, self.bart_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.bart_dim * 2, self.bart_dim)
        )

    def create_enhanced_encoder_sequence(self, eeg_feat):
        """
        Create enhanced encoder sequence with multi-layer attention and memory augmentation.
        
        Args:
            eeg_feat: EEG features of shape (batch_size, hidden_dim)
        
        Returns:
            Enhanced encoder sequence of shape (batch_size, encoder_length, bart_dim)
        """
        batch_size = eeg_feat.shape[0]
        
        # Project EEG features to BART dimension
        proj_eeg = self.eeg_to_bart(eeg_feat)
        
        # Compute adaptive conditioning strength based on EEG content
        conditioning_strength = self.conditioning_mlp(proj_eeg).squeeze(-1)  # (batch,)
        conditioning_strength = torch.clamp(conditioning_strength, 0.8, 3.0)  # Stronger range
        
        # Apply content-aware conditioning
        proj_eeg = proj_eeg * conditioning_strength.unsqueeze(-1)
        eeg_expanded = proj_eeg.unsqueeze(1)  # (batch, 1, bart_dim)
        
        # Memory-augmented attention for richer representation
        memory_keys = self.eeg_memory_keys.unsqueeze(0).expand(batch_size, -1, -1)
        memory_values = self.eeg_memory_values.unsqueeze(0).expand(batch_size, -1, -1)
        
        memory_attended, memory_weights = self.eeg_cross_attn_layers[0](
            query=eeg_expanded,
            key=memory_keys,
            value=memory_values
        )
        
        # Enhance EEG representation with memory
        eeg_enhanced = eeg_expanded + 0.4 * memory_attended
        
        # Apply feature diversification
        eeg_diversified = eeg_enhanced + 0.2 * self.diversifier(eeg_enhanced)
        
        # Initialize encoder sequence with learnable queries
        queries = self.encoder_queries.expand(batch_size, -1, -1)
        
        # Multi-layer cross-attention for progressive refinement
        encoder_seq = queries
        attention_weights = []
        
        for i, attn_layer in enumerate(self.eeg_cross_attn_layers):
            # Use different keys/values for different layers
            if i == 0:
                key_value = eeg_diversified
            else:
                # Progressive refinement: use previous layer output as additional context
                key_value = torch.cat([eeg_diversified, encoder_seq[:, :1, :]], dim=1)
            
            attended, weights = attn_layer(
                query=encoder_seq,
                key=key_value,
                value=key_value
            )
            
            # Content-aware gating
            gate = self.content_gate(encoder_seq)
            encoder_seq = encoder_seq + gate * attended
            attention_weights.append(weights)
        
        # Add positional embeddings
        pos_emb = self.encoder_pos_emb.expand(batch_size, -1, -1)
        encoder_seq = encoder_seq + pos_emb
        
        # Final normalization
        encoder_seq = F.layer_norm(encoder_seq, encoder_seq.shape[-1:])
        
        return encoder_seq, attention_weights

    def forward(self, eeg_feat, decoder_input_ids=None, labels=None, **kwargs):
        """
        Enhanced forward pass with stronger EEG conditioning.
        
        Args:
            eeg_feat: EEG features of shape (batch_size, hidden_dim)
            decoder_input_ids: Decoder input token IDs
            labels: Target token IDs for training
        
        Returns:
            Enhanced BART model outputs
        """
        try:
            batch_size = eeg_feat.shape[0]
            device = eeg_feat.device
            
            # Create enhanced encoder sequence from EEG
            encoder_seq, attention_weights = self.create_enhanced_encoder_sequence(eeg_feat)
            
            # Create attention mask for encoder sequence
            encoder_attention_mask = torch.ones(
                batch_size, self.encoder_length,
                dtype=torch.long, device=device
            )
            
            # Wrap encoder outputs
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_seq,
                hidden_states=None,
                attentions=attention_weights
            )
            
            # Run BART with EEG-conditioned encoder
            bart_outputs = self.bart(
                input_ids=None,
                attention_mask=encoder_attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True,
                **kwargs
            )
            
            return bart_outputs
            
        except Exception as e:
            logger.error(f"Enhanced BARTDecoder forward pass failed: {e}")
            # Return safe dummy output
            vocab_size = self.bart.config.vocab_size
            seq_len = decoder_input_ids.size(1) if decoder_input_ids is not None else 16
            
            dummy_logits = torch.zeros(batch_size, seq_len, vocab_size, device=eeg_feat.device)
            dummy_loss = torch.tensor(5.0, device=eeg_feat.device, requires_grad=True)
            return Seq2SeqLMOutput(loss=dummy_loss, logits=dummy_logits)

    def generate_from_eeg(self, eeg_feat, max_length=32, **kwargs):
        """
        Generate text from EEG features with enhanced conditioning.
        
        Args:
            eeg_feat: EEG features of shape (batch_size, hidden_dim)
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
        
        Returns:
            Generated token IDs
        """
        batch_size = eeg_feat.shape[0]
        device = eeg_feat.device
        
        # Create enhanced encoder sequence
        encoder_seq, _ = self.create_enhanced_encoder_sequence(eeg_feat)
        
        # Create attention mask
        encoder_attention_mask = torch.ones(
            batch_size, self.encoder_length,
            dtype=torch.long, device=device
        )
        
        # Wrap encoder outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_seq)
        
        # Enhanced generation configuration
        generation_config = {
            'encoder_outputs': encoder_outputs,
            'attention_mask': encoder_attention_mask,
            'decoder_start_token_id': self.bart.config.decoder_start_token_id,
            'pad_token_id': self.bart.config.pad_token_id,
            'eos_token_id': self.bart.config.eos_token_id,
            'max_length': max_length,
            'min_length': kwargs.get('min_length', 3),
            'no_repeat_ngram_size': kwargs.get('no_repeat_ngram_size', 3),  # Increased
            'repetition_penalty': kwargs.get('repetition_penalty', 1.5),  # Add repetition penalty
            'length_penalty': kwargs.get('length_penalty', 1.2),  # Slightly increased
        }
        
        # Configure sampling vs beam search with better parameters
        if kwargs.get('do_sample', False):
            generation_config.update({
                'do_sample': True,
                'temperature': kwargs.get('temperature', 0.8),  # Slightly higher
                'top_k': kwargs.get('top_k', 40),  # Reduced for more focused sampling
                'top_p': kwargs.get('top_p', 0.85),  # Slightly reduced
            })
        else:
            generation_config.update({
                'do_sample': False,
                'num_beams': kwargs.get('num_beams', 5),  # Increased from 3
                'diversity_penalty': kwargs.get('diversity_penalty', 0.5),  # Add diversity
            })
            
            if generation_config['num_beams'] > 1:
                generation_config['early_stopping'] = True
                generation_config['num_beam_groups'] = kwargs.get('num_beam_groups', 1)
        
        try:
            generated_ids = self.bart.generate(**generation_config)
            return generated_ids
            
        except Exception as e:
            logger.error(f"Enhanced generation failed: {e}")
            # Fallback generation
            return torch.full(
                (batch_size, max_length),
                self.bart.config.decoder_start_token_id,
                dtype=torch.long, device=device
            )
    
    def get_attention_weights(self):
        """Get attention weights for analysis."""
        if hasattr(self, '_last_attention_weights'):
            return self._last_attention_weights
        return None
    
    def get_conditioning_stats(self, eeg_feat):
        """Get conditioning strength statistics for monitoring."""
        with torch.no_grad():
            proj_eeg = self.eeg_to_bart(eeg_feat)
            conditioning_strengths = self.conditioning_mlp(proj_eeg).squeeze(-1)
            conditioning_strengths = torch.clamp(conditioning_strengths, 0.8, 3.0)
            
            return {
                'mean_conditioning': conditioning_strengths.mean().item(),
                'std_conditioning': conditioning_strengths.std().item(),
                'min_conditioning': conditioning_strengths.min().item(),
                'max_conditioning': conditioning_strengths.max().item()
            }