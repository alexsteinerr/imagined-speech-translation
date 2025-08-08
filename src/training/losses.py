"""
Enhanced composite loss for EEG-to-text model with anti-collapse mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple


class EnhancedCompositeSeq2SeqLoss(nn.Module):
    """
    Unified loss combining CE, alignment, BoW, diversity, and variance losses.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        pad_token_id: int,
        bow_indices: Optional[List[int]] = None,
        label_smoothing: float = 0.05,
        w_ce: float = 1.0,
        w_align: float = 0.5,
        w_bow: float = 0.2,
        w_div: float = 0.1,
        w_var: float = 0.05,
        tau: float = 0.07,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
        
        # Loss weights
        self.w_ce = w_ce
        self.w_align = w_align
        self.w_bow = w_bow
        self.w_div = w_div
        self.w_var = w_var
        self.tau = tau
        
        # Projection heads for alignment
        proj_dim = hidden_dim // 2
        self.eeg_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        self.txt_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # BoW head for important vocabulary
        if bow_indices is not None and len(bow_indices) > 0:
            self.register_buffer("bow_indices_tensor", torch.tensor(bow_indices, dtype=torch.long))
            self.bow_head = nn.Linear(hidden_dim, len(bow_indices))
        else:
            self.bow_indices_tensor = None
            self.bow_head = None

    def _label_smoothed_ce_loss(self, logits, labels, epsilon):
        """Compute label-smoothed cross-entropy loss."""
        B, T, V = logits.size()
        
        # Reshape for loss computation
        logits_flat = logits.reshape(-1, V)
        labels_flat = labels.reshape(-1)
        
        # Create smoothed target distribution
        smooth_targets = torch.zeros_like(logits_flat).scatter_(
            1, labels_flat.unsqueeze(1), 1.0 - epsilon
        )
        smooth_targets = smooth_targets + epsilon / V
        
        # Mask padding tokens
        mask = (labels_flat != self.pad_token_id) & (labels_flat != -100)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Compute KL divergence
        log_probs = F.log_softmax(logits_flat, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        # Apply mask and average
        loss = (loss * mask.float()).sum() / mask.float().sum()
        
        return loss

    def _alignment_loss(self, eeg_feats, dec_hidden, attention_mask):
        """Compute contrastive alignment loss between EEG and text representations."""
        B = eeg_feats.size(0)
        
        # Pool decoder hidden states
        if len(dec_hidden.shape) == 3:  # (B, T, H)
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            dec_pooled = (dec_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1.0)
        else:
            dec_pooled = dec_hidden
        
        # Project to common space
        z_eeg = F.normalize(self.eeg_proj(eeg_feats), dim=-1)
        z_txt = F.normalize(self.txt_proj(dec_pooled), dim=-1)
        
        # Compute similarity matrix
        sim = torch.matmul(z_eeg, z_txt.t()) / self.tau
        
        # Contrastive loss (InfoNCE style)
        labels = torch.arange(B, device=sim.device)
        loss_i = F.cross_entropy(sim, labels)
        loss_j = F.cross_entropy(sim.t(), labels)
        
        return 0.5 * (loss_i + loss_j)

    def _bow_loss(self, eeg_feats, labels):
        """Compute bag-of-words prediction loss from EEG features."""
        if self.bow_head is None or self.bow_indices_tensor is None:
            return None
        
        B, T = labels.size()
        device = labels.device
        
        # Create BoW targets
        bow_targets = torch.zeros(B, len(self.bow_indices_tensor), device=device)
        
        for b in range(B):
            valid_tokens = labels[b][(labels[b] != -100) & (labels[b] != self.pad_token_id)]
            if valid_tokens.numel() > 0:
                # Find which BoW indices are present
                for idx, vocab_id in enumerate(self.bow_indices_tensor):
                    if (valid_tokens == vocab_id).any():
                        bow_targets[b, idx] = 1.0
        
        # Predict BoW from EEG features
        bow_logits = self.bow_head(eeg_feats)
        
        # Binary cross-entropy loss
        return F.binary_cross_entropy_with_logits(bow_logits, bow_targets)

    def _diversity_loss(self, features):
        """Compute diversity loss to prevent feature collapse."""
        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Normalize features
        norm_features = F.normalize(features, dim=-1)
        
        # Compute pairwise similarities
        sim_matrix = torch.matmul(norm_features, norm_features.t())
        
        # Remove diagonal
        B = sim_matrix.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
        
        # Penalize high similarities (encourage diversity)
        off_diagonal = sim_matrix[mask]
        diversity_loss = off_diagonal.abs().mean()
        
        return diversity_loss

    def _variance_loss(self, features):
        """Encourage feature variance to prevent dimensional collapse."""
        # Compute variance across batch dimension
        feature_var = torch.var(features, dim=0)
        
        # Penalize low variance
        var_loss = torch.exp(-feature_var).mean()
        
        return var_loss

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_features: torch.Tensor,
        decoder_hidden: Optional[torch.Tensor] = None,
        return_components: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.
        
        Args:
            logits: Model output logits (B, T, V)
            labels: Target labels (B, T)
            attention_mask: Attention mask (B, T)
            encoder_features: EEG encoder features (B, H)
            decoder_hidden: Decoder hidden states (B, T, H) or None
            return_components: Whether to return individual loss components
        
        Returns:
            Dictionary containing total loss and optionally individual components
        """
        # Cross-entropy loss (always computed)
        ce_loss = self._label_smoothed_ce_loss(logits, labels, self.label_smoothing)
        total_loss = self.w_ce * ce_loss
        
        components = {'loss_ce': ce_loss.detach()}
        
        # Alignment loss (if decoder hidden states available)
        if decoder_hidden is not None and self.w_align > 0:
            align_loss = self._alignment_loss(encoder_features, decoder_hidden, attention_mask)
            total_loss = total_loss + self.w_align * align_loss
            components['loss_align'] = align_loss.detach()
        
        # BoW loss (if configured)
        if self.bow_head is not None and self.w_bow > 0:
            bow_loss = self._bow_loss(encoder_features, labels)
            if bow_loss is not None:
                total_loss = total_loss + self.w_bow * bow_loss
                components['loss_bow'] = bow_loss.detach()
        
        # Diversity loss
        if self.w_div > 0:
            div_loss = self._diversity_loss(encoder_features)
            total_loss = total_loss + self.w_div * div_loss
            components['loss_div'] = div_loss.detach()
        
        # Variance loss
        if self.w_var > 0:
            var_loss = self._variance_loss(encoder_features)
            total_loss = total_loss + self.w_var * var_loss
            components['loss_var'] = var_loss.detach()
        
        if return_components:
            return {
                'loss': total_loss,
                **components
            }
        else:
            return {'loss': total_loss}


def get_top_k_vocab_indices(tokenizer, k: int = 3000) -> List[int]:
    """
    Get top-k most important vocabulary indices for BoW loss.
    
    Args:
        tokenizer: Tokenizer instance
        k: Number of top vocabulary items to consider
    
    Returns:
        List of vocabulary indices
    """
    # Get special token IDs to exclude
    special_ids = set()
    for attr in ['pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id', 'sep_token_id', 'cls_token_id']:
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(token_id)
    
    # For Chinese tokenizer, you might want to prioritize common characters
    # This is a simple heuristic - ideally use corpus frequency
    vocab = tokenizer.get_vocab()
    
    # Filter out special tokens and sort by token ID (rough frequency proxy)
    valid_ids = []
    for token, idx in vocab.items():
        if idx not in special_ids and not token.startswith('##'):
            valid_ids.append(idx)
    
    # Return first k valid IDs
    return valid_ids[:min(k, len(valid_ids))]


class AdaptiveLossScheduler:
    """
    Adaptive scheduler for loss weights based on training dynamics.
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adaptation_rate: float = 0.01,
        min_weights: Optional[Dict[str, float]] = None,
        max_weights: Optional[Dict[str, float]] = None
    ):
        self.weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.min_weights = min_weights or {k: 0.01 for k in initial_weights}
        self.max_weights = max_weights or {k: 2.0 for k in initial_weights}
        
        # Track loss history for adaptation
        self.loss_history = {k: [] for k in initial_weights}
        self.window_size = 10
    
    def update(self, loss_components: Dict[str, float], diversity_score: float = None):
        """
        Update loss weights based on current training dynamics.
        
        Args:
            loss_components: Current loss component values
            diversity_score: Current diversity score (0-1, higher is better)
        """
        # Track loss history
        for key in self.weights:
            if f'loss_{key}' in loss_components:
                self.loss_history[key].append(loss_components[f'loss_{key}'])
                if len(self.loss_history[key]) > self.window_size:
                    self.loss_history[key].pop(0)
        
        # Adapt weights based on diversity score
        if diversity_score is not None:
            if diversity_score < 0.3:  # Low diversity - increase diversity/variance weights
                self.weights['div'] = min(
                    self.weights['div'] * (1 + self.adaptation_rate),
                    self.max_weights['div']
                )
                self.weights['var'] = min(
                    self.weights['var'] * (1 + self.adaptation_rate),
                    self.max_weights['var']
                )
            elif diversity_score > 0.8:  # High diversity - can reduce diversity weights
                self.weights['div'] = max(
                    self.weights['div'] * (1 - self.adaptation_rate),
                    self.min_weights['div']
                )
        
        # Balance alignment weight based on CE loss trend
        if len(self.loss_history['ce']) >= 5:
            recent_ce = self.loss_history['ce'][-5:]
            if all(recent_ce[i] <= recent_ce[i+1] for i in range(4)):  # CE increasing
                # Increase alignment to help learning
                self.weights['align'] = min(
                    self.weights['align'] * (1 + self.adaptation_rate * 0.5),
                    self.max_weights['align']
                )
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.weights.copy()