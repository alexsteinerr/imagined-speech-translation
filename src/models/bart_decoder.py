"""
BART-based decoder for converting EEG features to text.
"""

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import logging

logger = logging.getLogger(__name__)


class BARTDecoder(nn.Module):
    """
    BART decoder that converts EEG features to text sequences.
    """
    
    def __init__(self, hidden_dim, disable_cross_modal=False):
        super().__init__()
        self.disable_cross_modal = disable_cross_modal
        self.hidden_dim = hidden_dim
        
        # Load pretrained BART
        self.bart = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
        self.bart_dim = self.bart.config.d_model
        
        # EEG to BART feature projection
        self.eeg_to_bart = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.bart_dim),
            nn.LayerNorm(self.bart_dim)
        )
        
        # Conservative initialization to prevent gradient explosion
        for layer in self.eeg_to_bart:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Encoder sequence parameters
        self.encoder_length = 8
        self.eeg_conditioning_strength = nn.Parameter(torch.tensor(0.2))
        self.encoder_queries = nn.Parameter(torch.randn(1, self.encoder_length, self.bart_dim) * 0.01)
        
        # Cross-attention for EEG conditioning
        self.eeg_attention = nn.MultiheadAttention(
            embed_dim=self.bart_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Positional embeddings
        self.encoder_pos_emb = nn.Parameter(torch.randn(1, self.encoder_length, self.bart_dim) * 0.01)

    def create_encoder_sequence(self, eeg_feat):
        """
        Create encoder sequence from EEG features.
        
        Args:
            eeg_feat: EEG features of shape (batch_size, hidden_dim)
        
        Returns:
            Encoder sequence of shape (batch_size, encoder_length, bart_dim)
        """
        batch_size = eeg_feat.shape[0]
        
        # Apply conditioning strength with clamping
        conditioning_strength = torch.clamp(self.eeg_conditioning_strength, 0.05, 1.0)
        proj_eeg = self.eeg_to_bart(eeg_feat) * conditioning_strength
        eeg_expanded = proj_eeg.unsqueeze(1)
        
        # Generate encoder sequence using attention
        queries = self.encoder_queries.expand(batch_size, -1, -1)
        encoder_seq, _ = self.eeg_attention(
            query=queries,
            key=eeg_expanded,
            value=eeg_expanded
        )
        
        # Add positional embeddings
        encoder_seq = encoder_seq + self.encoder_pos_emb
        return encoder_seq

    def forward(self, eeg_feat, decoder_input_ids=None, labels=None, **kwargs):
        """
        Forward pass through the decoder.
        
        Args:
            eeg_feat: EEG features of shape (batch_size, hidden_dim)
            decoder_input_ids: Decoder input token IDs
            labels: Target token IDs for training
        
        Returns:
            BART model outputs
        """
        try:
            batch_size = eeg_feat.shape[0]
            device = eeg_feat.device
            
            # Create encoder sequence from EEG
            encoder_seq = self.create_encoder_sequence(eeg_feat)
            encoder_attention_mask = torch.ones(
                batch_size, self.encoder_length,
                dtype=torch.long, device=device
            )
            
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_seq)
            
            # Run BART with EEG-conditioned encoder
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
            # Return safe dummy output
            vocab_size = self.bart.config.vocab_size
            seq_len = decoder_input_ids.size(1) if decoder_input_ids is not None else 16
            
            dummy_logits = torch.zeros(batch_size, seq_len, vocab_size, device=eeg_feat.device)
            dummy_loss = torch.tensor(5.0, device=eeg_feat.device, requires_grad=True)
            return Seq2SeqLMOutput(loss=dummy_loss, logits=dummy_logits)

    def generate_from_eeg(self, eeg_feat, max_length=32, **kwargs):
        """
        Generate text from EEG features.
        
        Args:
            eeg_feat: EEG features of shape (batch_size, hidden_dim)
            max_length: Maximum generation length
            **kwargs: Additional generation parameters
        
        Returns:
            Generated token IDs
        """
        batch_size = eeg_feat.shape[0]
        device = eeg_feat.device
        
        # Create encoder sequence
        encoder_seq = self.create_encoder_sequence(eeg_feat)
        encoder_attention_mask = torch.ones(
            batch_size, self.encoder_length,
            dtype=torch.long, device=device
        )
        
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_seq)
        
        # Setup generation configuration
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
        
        # Configure sampling vs beam search
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