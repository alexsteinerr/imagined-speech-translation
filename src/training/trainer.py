"""
Enhanced training loop and utilities for EEG-to-text model with diversity losses.
"""

import os
import time
import torch
import torch.nn.functional as F
import wandb
import numpy as np
from tqdm.auto import tqdm
from ..evaluation.evaluator import ChineseEvaluator
import logging

logger = logging.getLogger(__name__)


class EEGTrainer:
    """
    Enhanced trainer class for EEG-to-text model with anti-collapse mechanisms.
    """
    
    def __init__(self, model, tokenizer, train_loader, val_loader, optimizer, scheduler, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        self.device = next(model.parameters()).device
        self.evaluator = ChineseEvaluator()
        
        # Training state
        self.best_bleu4 = 0.0
        self.patience_counter = 0
        self.global_step = 0
        
        # Enhanced loss weights
        self.diversity_loss_weight = config.get('diversity_loss_weight', 0.1)
        self.alignment_loss_weight = config.get('alignment_loss_weight', 0.05)
        self.feature_variance_weight = config.get('feature_variance_weight', 0.02)
        
        # Anti-collapse monitoring
        self.repetition_threshold = 0.8  # Threshold for detecting repetitive outputs
        self.diversity_history = []
        
        logger.info(f"Enhanced trainer initialized with diversity_loss_weight={self.diversity_loss_weight}")

    def compute_diversity_loss(self, features):
        """
        Compute diversity loss to prevent feature collapse.
        Encourages features to be different across samples.
        """
        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Normalize features
        norm_features = F.normalize(features, dim=-1)
        
        # Compute pairwise similarities
        similarity_matrix = torch.matmul(norm_features, norm_features.transpose(-2, -1))
        
        # Remove diagonal (self-similarities)
        batch_size = similarity_matrix.size(0)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
        
        # Penalize high similarities between different samples
        off_diagonal_similarities = similarity_matrix[mask]
        diversity_loss = off_diagonal_similarities.abs().mean()
        
        return diversity_loss

    def compute_feature_variance_loss(self, features):
        """
        Encourage feature variance to prevent dimensional collapse.
        """
        # Compute variance across batch dimension
        feature_var = torch.var(features, dim=0, keepdim=False)
        
        # Penalize low variance (encourage each dimension to be used)
        variance_loss = torch.exp(-feature_var).mean()
        
        return variance_loss

    def compute_alignment_loss(self, eeg_features, decoder_hidden_states):
        """
        Compute alignment loss between EEG features and decoder states.
        This encourages the EEG features to be relevant for text generation.
        """
        if decoder_hidden_states is None or decoder_hidden_states.size(0) != eeg_features.size(0):
            return torch.tensor(0.0, device=eeg_features.device, requires_grad=True)
        
        # Use the first decoder hidden state as text representation
        if len(decoder_hidden_states.shape) > 2:
            text_repr = decoder_hidden_states[:, 0, :]  # First token representation
        else:
            text_repr = decoder_hidden_states
        
        # Ensure same dimensionality
        if text_repr.size(-1) != eeg_features.size(-1):
            text_repr = F.linear(text_repr, 
                               torch.randn(eeg_features.size(-1), text_repr.size(-1), 
                                         device=eeg_features.device) * 0.01)
        
        # Normalize both representations
        eeg_norm = F.normalize(eeg_features, dim=-1)
        text_norm = F.normalize(text_repr, dim=-1)
        
        # Compute cosine similarity and encourage positive alignment
        alignment = torch.sum(eeg_norm * text_norm, dim=-1)
        alignment_loss = -alignment.mean()  # Negative because we want to maximize alignment
        
        return alignment_loss

    def detect_repetitive_generation(self, predictions):
        """
        Detect if the model is generating repetitive outputs.
        """
        if not predictions or len(predictions) < 2:
            return False
        
        # Check if too many predictions are identical
        unique_predictions = set(predictions)
        repetition_rate = 1.0 - (len(unique_predictions) / len(predictions))
        
        return repetition_rate > self.repetition_threshold

    def enhanced_forward_pass(self, eeg, decoder_input_ids, labels):
        """Enhanced forward pass with anti-collapse losses."""
        try:
            vocab_size = len(self.tokenizer.get_vocab())
            
            # Validate token IDs
            if decoder_input_ids.max() >= vocab_size:
                return None
            if (labels != -100).any() and labels[labels != -100].max() >= vocab_size:
                return None
            
            # Get EEG features from brain encoder
            eeg_features = self.model.brain_encoder(eeg)
            
            # Forward through decoder
            outputs = self.model.bart_decoder(
                eeg_feat=eeg_features,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                label_smoothing_factor=self.config.get('label_smoothing', 0.1),
                return_dict=True
            )
            
            if outputs.loss is None:
                return None
                
            # Start with base loss
            total_loss = outputs.loss
            loss_components = {'base_loss': outputs.loss.item()}
            
            # Add diversity loss to prevent feature collapse
            diversity_loss = self.compute_diversity_loss(eeg_features)
            total_loss = total_loss + self.diversity_loss_weight * diversity_loss
            loss_components['diversity_loss'] = diversity_loss.item()
            
            # Add feature variance loss
            variance_loss = self.compute_feature_variance_loss(eeg_features)
            total_loss = total_loss + self.feature_variance_weight * variance_loss
            loss_components['variance_loss'] = variance_loss.item()
            
            # Add alignment loss if decoder hidden states are available
            if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
                alignment_loss = self.compute_alignment_loss(eeg_features, outputs.decoder_hidden_states[-1])
                total_loss = total_loss + self.alignment_loss_weight * alignment_loss
                loss_components['alignment_loss'] = alignment_loss.item()
            
            # Store loss components for logging
            outputs.loss = total_loss
            outputs.loss_components = loss_components
            
            return outputs
            
        except RuntimeError as e:
            if "CUDA error" in str(e) or "out of memory" in str(e):
                torch.cuda.empty_cache()
                return None
            else:
                raise e

    def safe_forward_pass(self, eeg, decoder_input_ids, labels):
        """Legacy wrapper for compatibility."""
        return self.enhanced_forward_pass(eeg, decoder_input_ids, labels)

    def train_epoch(self, epoch):
        """Enhanced training epoch with diversity monitoring."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        accumulation_step = 0
        
        # Track loss components
        loss_components_sum = {}
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            try:
                eeg = [region.to(self.device) for region in batch['eeg']]
                decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.enhanced_forward_pass(eeg, decoder_input_ids, labels)
                
                if outputs is None or outputs.loss is None:
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                
                loss = outputs.loss / self.config['accumulation_steps']
                
                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad()
                    continue
                
                loss.backward()
                accumulation_step += 1
                
                # Accumulate loss components for logging
                if hasattr(outputs, 'loss_components'):
                    for component, value in outputs.loss_components.items():
                        if component not in loss_components_sum:
                            loss_components_sum[component] = 0.0
                        loss_components_sum[component] += value
                
                # Update every accumulation_steps
                if accumulation_step >= self.config['accumulation_steps']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip_norm']
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    
                    accumulation_step = 0
                    self.global_step += 1
                
                total_loss += outputs.loss.item() * len(labels)
                total_samples += len(labels)
                
                # Enhanced logging with loss components
                if step % self.config['log_interval'] == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    postfix = {
                        'loss': f"{outputs.loss.item():.4f}",
                        'lr': f"{current_lr:.1e}"
                    }
                    
                    if hasattr(outputs, 'loss_components'):
                        for component, value in outputs.loss_components.items():
                            if component != 'base_loss':
                                postfix[component[:3]] = f"{value:.3f}"
                    
                    pbar.set_postfix(postfix)
                    
                    # Log to wandb with components
                    log_dict = {
                        "train/step_loss": outputs.loss.item(),
                        "train/lr": current_lr,
                        "step": self.global_step
                    }
                    
                    if hasattr(outputs, 'loss_components'):
                        for component, value in outputs.loss_components.items():
                            log_dict[f"train/{component}"] = value
                    
                    wandb.log(log_dict)
                    
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA error" in str(e):
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue
                else:
                    raise e
        
        # Handle remaining gradients
        if accumulation_step > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        # Log average loss components for the epoch
        if loss_components_sum and total_samples > 0:
            for component, total_value in loss_components_sum.items():
                avg_value = total_value / (total_samples / self.config['batch_size'])
                wandb.log({f"train/epoch_{component}": avg_value, "epoch": epoch})
        
        return avg_loss

    def evaluate(self):
        """Enhanced evaluation with repetition detection."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_val_loss = 0.0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                try:
                    eeg = [region.to(self.device) for region in batch['eeg']]
                    decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Calculate validation loss with enhanced forward pass
                    val_outputs = self.enhanced_forward_pass(eeg, decoder_input_ids, labels)
                    if val_outputs is not None and val_outputs.loss is not None:
                        total_val_loss += val_outputs.loss.item() * len(labels)
                        total_val_samples += len(labels)
                    
                    # Generate predictions with enhanced parameters
                    gen_config = {
                        'max_length': self.config['max_gen_length'],
                        'num_beams': self.config['num_beams'],
                        'length_penalty': self.config['length_penalty'],
                        'no_repeat_ngram_size': max(3, self.config.get('no_repeat_ngram_size', 2)),
                        'repetition_penalty': 1.5,  # Add repetition penalty
                        'min_length': self.config['min_length'],
                        'do_sample': False,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'diversity_penalty': 0.5  # Encourage diverse beam search
                    }
                    
                    if self.config['num_beams'] > 1:
                        gen_config['early_stopping'] = True
                        gen_config['num_beam_groups'] = min(2, self.config['num_beams'])
                    
                    generated_ids = self.model.generate(eeg_data=eeg, **gen_config)
                    
                    for i in range(len(generated_ids)):
                        pred_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                        all_predictions.append(pred_text.strip())
                        
                        target_ids = labels[i][labels[i] != -100]
                        target_text = self.tokenizer.decode(target_ids, skip_special_tokens=True)
                        all_targets.append(target_text.strip())
                            
                except Exception:
                    torch.cuda.empty_cache()
                    continue
        
        # Calculate average validation loss
        avg_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else float('inf')
        
        if not all_predictions:
            return self._empty_metrics_with_loss(avg_val_loss)
        
        # Detect repetitive generation
        is_repetitive = self.detect_repetitive_generation(all_predictions)
        
        try:
            metrics = self.evaluator.compute_all_metrics(all_predictions, all_targets)
            metrics['val_loss'] = avg_val_loss
            metrics['predictions'] = all_predictions
            metrics['targets'] = all_targets
            metrics['is_repetitive'] = is_repetitive
            
            # Compute diversity score
            unique_predictions = len(set(all_predictions))
            diversity_score = unique_predictions / len(all_predictions) if all_predictions else 0.0
            metrics['diversity_score'] = diversity_score
            
            # Store diversity history
            self.diversity_history.append(diversity_score)
            if len(self.diversity_history) > 10:
                self.diversity_history.pop(0)
            
            # Log warning if repetitive
            if is_repetitive:
                logger.warning(f"Repetitive generation detected! Diversity score: {diversity_score:.3f}")
                
        except Exception as e:
            logger.error(f"Evaluation metrics computation failed: {e}")
            metrics = self._empty_metrics_with_loss(avg_val_loss)
            metrics['is_repetitive'] = is_repetitive
            metrics['diversity_score'] = 0.0
        
        return metrics

    def _empty_metrics_with_loss(self, val_loss):
        """Return empty metrics with loss information."""
        return {
            'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 
            'rouge_l_f': 0.0, 'val_loss': val_loss,
            'predictions': [], 'targets': [], 'is_repetitive': True, 'diversity_score': 0.0
        }

    def log_predictions_to_wandb(self, predictions, targets, epoch, max_samples=10):
        """Enhanced prediction logging with repetition analysis."""
        num_samples = min(len(predictions), max_samples)
        
        # Create data for wandb table
        table_data = []
        repetitive_count = 0
        
        for i in range(num_samples):
            pred = predictions[i]
            target = targets[i]
            
            # Check if this prediction is repetitive
            is_repeat = predictions.count(pred) > 1
            if is_repeat:
                repetitive_count += 1
            
            table_data.append([
                i + 1,
                target,
                pred,
                len(target.split()),
                len(pred.split()),
                "Yes" if is_repeat else "No"
            ])
        
        # Create wandb table
        table = wandb.Table(
            columns=["Sample", "Target", "Prediction", "Target_Length", "Pred_Length", "Repetitive"],
            data=table_data
        )
        
        wandb.log({
            f"predictions_table_epoch_{epoch}": table,
            f"repetitive_predictions_epoch_{epoch}": repetitive_count,
            "epoch": epoch
        })

    def log_enhanced_metrics(self, metrics, epoch):
        """Log enhanced metrics including diversity and repetition detection."""
        log_dict = {
            "val/loss": metrics['val_loss'],
            "val/bleu_1": metrics['bleu_1'],
            "val/bleu_2": metrics['bleu_2'],
            "val/bleu_3": metrics['bleu_3'],
            "val/bleu_4": metrics['bleu_4'],
            "val/rouge_l": metrics['rouge_l_f'],
            "val/diversity_score": metrics['diversity_score'],
            "val/is_repetitive": float(metrics['is_repetitive']),
            "epoch": epoch
        }
        
        # Add diversity trend
        if len(self.diversity_history) > 1:
            diversity_trend = self.diversity_history[-1] - self.diversity_history[-2]
            log_dict["val/diversity_trend"] = diversity_trend
        
        wandb.log(log_dict)

    def log_region_weights_to_wandb(self, prefix="", step=None):
        """Enhanced region weights logging with diversity info."""
        if not hasattr(self.model, 'brain_encoder'):
            return
        
        try:
            weights_info = self.model.brain_encoder.get_region_weights()
            log_dict = {}
            
            # Log individual region weights
            for name, weight in zip(weights_info['names'], weights_info['softmax']):
                log_dict[f"{prefix}region_weights/{name}"] = float(weight)
            
            # Log weight entropy (diversity measure)
            weights_array = np.array(weights_info['softmax'])
            weight_entropy = -np.sum(weights_array * np.log(weights_array + 1e-8))
            log_dict[f"{prefix}region_weight_entropy"] = weight_entropy
            
            # Log raw importance if available
            if hasattr(self.model.brain_encoder, 'region_importance') and not self.model.brain_encoder.uniform_region_weight:
                raw_weights = self.model.brain_encoder.region_importance.data.cpu().numpy()
                for name, raw_weight in zip(weights_info['names'], raw_weights):
                    log_dict[f"{prefix}region_importance_raw/{name}"] = float(raw_weight)
            
            if step is not None:
                log_dict["step"] = step
            
            wandb.log(log_dict)
            
        except Exception as e:
            logger.warning(f"Failed to log enhanced region weights: {e}")

    def save_checkpoint(self, epoch, metrics, save_path):
        """Enhanced checkpoint saving with training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'global_step': self.global_step,
            'best_bleu4': self.best_bleu4,
            'diversity_history': self.diversity_history,
            'loss_weights': {
                'diversity_loss_weight': self.diversity_loss_weight,
                'alignment_loss_weight': self.alignment_loss_weight,
                'feature_variance_weight': self.feature_variance_weight
            }
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Enhanced checkpoint saved: {save_path}")

    def train(self):
        """Enhanced main training loop with anti-collapse monitoring."""
        logger.info(f"Starting enhanced training for {self.config['epochs']} epochs...")
        logger.info(f"Loss weights - Diversity: {self.diversity_loss_weight}, "
                   f"Alignment: {self.alignment_loss_weight}, Variance: {self.feature_variance_weight}")
        
        consecutive_repetitive_epochs = 0
        max_consecutive_repetitive = 3
        
        for epoch in range(self.config['epochs']):
            # Training
            train_loss = self.train_epoch(epoch)
            
            wandb.log({
                "train/epoch_loss": train_loss,
                "epoch": epoch
            })
            
            # Log region weights during training
            self.log_region_weights_to_wandb(prefix="train/", step=self.global_step)
            
            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                val_metrics = self.evaluate()
                bleu4 = val_metrics['bleu_4']
                is_repetitive = val_metrics['is_repetitive']
                diversity_score = val_metrics['diversity_score']
                
                # Enhanced metrics logging
                self.log_enhanced_metrics(val_metrics, epoch)
                self.log_region_weights_to_wandb(prefix="val/", step=self.global_step)
                
                # Log prediction table
                if val_metrics['predictions'] and val_metrics['targets']:
                    self.log_predictions_to_wandb(
                        val_metrics['predictions'], 
                        val_metrics['targets'], 
                        epoch
                    )
                
                # Anti-collapse monitoring
                if is_repetitive:
                    consecutive_repetitive_epochs += 1
                    logger.warning(f"Repetitive generation detected for {consecutive_repetitive_epochs} consecutive evaluations")
                    
                    # Adjust loss weights if too repetitive
                    if consecutive_repetitive_epochs >= 2:
                        self.diversity_loss_weight = min(0.3, self.diversity_loss_weight * 1.5)
                        self.feature_variance_weight = min(0.1, self.feature_variance_weight * 1.3)
                        logger.info(f"Increased diversity loss weight to {self.diversity_loss_weight:.3f}")
                        logger.info(f"Increased variance loss weight to {self.feature_variance_weight:.3f}")
                else:
                    consecutive_repetitive_epochs = 0
                    # Gradually reduce diversity weight if generation is diverse
                    if diversity_score > 0.7:
                        self.diversity_loss_weight = max(0.05, self.diversity_loss_weight * 0.95)
                
                # Save best model
                if bleu4 > self.best_bleu4 and not is_repetitive:
                    self.best_bleu4 = bleu4
                    save_path = os.path.join(self.config['save_dir'], "best_model.pth")
                    checkpoint_metrics = {k: v for k, v in val_metrics.items() 
                                        if k not in ['predictions', 'targets']}
                    self.save_checkpoint(epoch, checkpoint_metrics, save_path)
                    logger.info(f"New best BLEU-4: {bleu4:.3f} (diverse generation)")
                    self.patience_counter = 0
                elif not is_repetitive:
                    self.patience_counter += 1
                else:
                    # Don't increase patience for repetitive generations
                    logger.info(f"Skipping patience increment due to repetitive generation")
                
                # Early stopping with repetition consideration
                if consecutive_repetitive_epochs >= max_consecutive_repetitive:
                    logger.warning(f"Stopping due to {max_consecutive_repetitive} consecutive repetitive generations")
                    break
                    
                if self.patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Regular checkpoints
            if (epoch + 1) % self.config['save_every_n_epochs'] == 0:
                save_path = os.path.join(self.config['save_dir'], f"checkpoint_epoch_{epoch+1}.pth")
                self.save_checkpoint(epoch, {}, save_path)
        
        logger.info(f"Enhanced training completed. Best BLEU-4: {self.best_bleu4:.3f}")
        return self.best_bleu4