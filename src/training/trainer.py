"""
Enhanced training loop for EEG-to-text model with composite loss.
"""

import os
import time
import torch
import wandb
import numpy as np
from tqdm.auto import tqdm
import logging

from ..evaluation.evaluator import ChineseEvaluator
from .losses import EnhancedCompositeSeq2SeqLoss, get_top_k_vocab_indices, AdaptiveLossScheduler

logger = logging.getLogger(__name__)


class EEGTrainer:
    """
    Trainer class for EEG-to-text model with composite loss and anti-collapse mechanisms.
    """
    
    def __init__(self, model, tokenizer, train_loader, val_loader, optimizer, scheduler, config):
        """
        Initialize trainer with composite loss.
        
        Args:
            model: EEG-to-text model
            tokenizer: Text tokenizer
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Training configuration dictionary
        """
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
        self.best_diversity = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self.epoch = 0
        
        # Diversity tracking
        self.diversity_history = []
        self.repetition_history = []
        
        # Initialize composite loss
        self._initialize_composite_loss()
        
        # Initialize adaptive scheduler if enabled
        if config.get('use_adaptive_loss', True):
            self._initialize_adaptive_scheduler()
        else:
            self.adaptive_scheduler = None
        
        logger.info(f"Trainer initialized with composite loss")
        logger.info(f"Loss weights: {self.config['loss_weights']}")

    def _initialize_composite_loss(self):
        """Initialize the composite loss function."""
        # Get BoW indices
        bow_vocab_size = self.config.get('bow_vocab_size', 2000)
        bow_indices = get_top_k_vocab_indices(self.tokenizer, k=bow_vocab_size)
        
        # Create composite loss
        loss_weights = self.config['loss_weights']
        self.composite_loss = EnhancedCompositeSeq2SeqLoss(
            vocab_size=len(self.tokenizer.get_vocab()),
            hidden_dim=self.config['hidden_dim'],
            pad_token_id=self.tokenizer.pad_token_id,
            bow_indices=bow_indices,
            label_smoothing=self.config.get('label_smoothing', 0.05),
            w_ce=loss_weights['ce'],
            w_align=loss_weights['align'],
            w_bow=loss_weights['bow'],
            w_div=loss_weights['div'],
            w_var=loss_weights['var'],
            tau=self.config.get('contrastive_tau', 0.07)
        ).to(self.device)
        
        logger.info(f"Composite loss initialized with {len(bow_indices)} BoW indices")

    def _initialize_adaptive_scheduler(self):
        """Initialize adaptive loss weight scheduler."""
        self.adaptive_scheduler = AdaptiveLossScheduler(
            initial_weights=self.config['loss_weights'].copy(),
            adaptation_rate=self.config.get('adaptation_rate', 0.01),
            min_weights={
                'ce': 1.0,      # CE always stays at 1.0
                'align': 0.2,   # Minimum alignment
                'bow': 0.05,    # Minimum BoW
                'div': 0.02,    # Minimum diversity
                'var': 0.01     # Minimum variance
            },
            max_weights={
                'ce': 1.0,      # CE always stays at 1.0
                'align': 0.8,   # Maximum alignment
                'bow': 0.3,     # Maximum BoW
                'div': 0.3,     # Maximum diversity
                'var': 0.15     # Maximum variance
            }
        )

    def forward_pass(self, eeg, decoder_input_ids, labels):
        """
        Forward pass with composite loss computation.
        
        Args:
            eeg: List of EEG tensors for each brain region
            decoder_input_ids: Decoder input token IDs
            labels: Target labels for loss computation
            
        Returns:
            Model outputs with loss and components
        """
        try:
            # Validate inputs
            vocab_size = len(self.tokenizer.get_vocab())
            if decoder_input_ids.max() >= vocab_size or (labels != -100).any() and labels[labels != -100].max() >= vocab_size:
                logger.warning("Invalid token IDs detected, skipping batch")
                return None
            
            # Get EEG features
            eeg_features = self.model.brain_encoder(eeg)
            
            # Forward through decoder (should return decoder_hidden if properly modified)
            outputs = self.model.bart_decoder(
                eeg_feat=eeg_features,
                decoder_input_ids=decoder_input_ids,
                labels=None,  # We'll compute loss separately
                return_dict=True
            )
            
            # Extract components
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs.get('logits')
            decoder_hidden = outputs.decoder_hidden if hasattr(outputs, 'decoder_hidden') else None
            
            # Create attention mask
            attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id).float()
            
            # Compute composite loss
            loss_dict = self.composite_loss(
                logits=logits,
                labels=labels,
                attention_mask=attention_mask,
                encoder_features=eeg_features,
                decoder_hidden=decoder_hidden,
                return_components=True
            )
            
            # Package outputs
            outputs.loss = loss_dict['loss']
            outputs.loss_components = {k: v.item() if torch.is_tensor(v) else v 
                                      for k, v in loss_dict.items() if k != 'loss'}
            outputs.eeg_features = eeg_features
            
            return outputs
            
        except RuntimeError as e:
            if "CUDA error" in str(e) or "out of memory" in str(e):
                logger.warning(f"CUDA error in forward pass: {e}")
                torch.cuda.empty_cache()
                return None
            else:
                raise e

    def train_epoch(self, epoch):
        """
        Train for one epoch with composite loss.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        self.epoch = epoch
        
        total_loss = 0.0
        total_samples = 0
        accumulation_step = 0
        
        # Track loss components
        loss_components_sum = {
            'loss_ce': 0.0,
            'loss_align': 0.0,
            'loss_bow': 0.0,
            'loss_div': 0.0,
            'loss_var': 0.0
        }
        component_counts = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            try:
                # Move data to device
                eeg = [region.to(self.device) for region in batch['eeg']]
                decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with composite loss
                outputs = self.forward_pass(eeg, decoder_input_ids, labels)
                
                if outputs is None or outputs.loss is None:
                    self.optimizer.zero_grad()
                    continue
                
                # Scale loss for gradient accumulation
                loss = outputs.loss / self.config['accumulation_steps']
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at step {step}")
                    self.optimizer.zero_grad()
                    continue
                
                # Backward pass
                loss.backward()
                accumulation_step += 1
                
                # Accumulate loss components
                if hasattr(outputs, 'loss_components'):
                    for key, value in outputs.loss_components.items():
                        if key in loss_components_sum:
                            loss_components_sum[key] += value
                    component_counts += 1
                
                # Optimizer step
                if accumulation_step >= self.config['accumulation_steps']:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('grad_clip_norm', 1.0)
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    
                    accumulation_step = 0
                    self.global_step += 1
                
                # Track total loss
                total_loss += outputs.loss.item() * len(labels)
                total_samples += len(labels)
                
                # Logging
                if step % self.config['log_interval'] == 0 and step > 0:
                    self._log_training_step(outputs, pbar)
                    
            except Exception as e:
                logger.error(f"Error in training step {step}: {e}")
                self.optimizer.zero_grad()
                continue
        
        # Final gradient step if needed
        if accumulation_step > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('grad_clip_norm', 1.0)
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Calculate epoch averages
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        # Log epoch summary
        if component_counts > 0:
            epoch_components = {k: v / component_counts for k, v in loss_components_sum.items()}
            self._log_epoch_summary(epoch, avg_loss, epoch_components)
        
        return avg_loss

    def evaluate(self):
        """
        Evaluate model on validation set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_val_loss = 0.0
        total_samples = 0
        
        # Loss component tracking
        loss_components_sum = {
            'loss_ce': 0.0,
            'loss_align': 0.0,
            'loss_bow': 0.0,
            'loss_div': 0.0,
            'loss_var': 0.0
        }
        component_counts = 0
        
        # Get generation config for evaluation
        gen_config = self.config['generation']['eval'].copy()
        gen_config['pad_token_id'] = self.tokenizer.pad_token_id
        gen_config['eos_token_id'] = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                try:
                    # Move data to device
                    eeg = [region.to(self.device) for region in batch['eeg']]
                    decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Calculate validation loss
                    val_outputs = self.forward_pass(eeg, decoder_input_ids, labels)
                    
                    if val_outputs is not None and val_outputs.loss is not None:
                        total_val_loss += val_outputs.loss.item() * len(labels)
                        total_samples += len(labels)
                        
                        # Track loss components
                        if hasattr(val_outputs, 'loss_components'):
                            for key, value in val_outputs.loss_components.items():
                                if key in loss_components_sum:
                                    loss_components_sum[key] += value
                            component_counts += 1
                    
                    # Generate predictions
                    generated_ids = self.model.generate(eeg_data=eeg, **gen_config)
                    
                    # Decode predictions and targets
                    for i in range(len(generated_ids)):
                        pred_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                        all_predictions.append(pred_text.strip())
                        
                        target_ids = labels[i][labels[i] != -100]
                        target_text = self.tokenizer.decode(target_ids, skip_special_tokens=True)
                        all_targets.append(target_text.strip())
                        
                except Exception as e:
                    logger.error(f"Error in evaluation: {e}")
                    continue
        
        # Calculate metrics
        metrics = self._compute_metrics(
            all_predictions, 
            all_targets, 
            total_val_loss, 
            total_samples,
            loss_components_sum,
            component_counts
        )
        
        # Update diversity tracking
        self._update_diversity_tracking(metrics)
        
        return metrics

    def _compute_metrics(self, predictions, targets, total_loss, total_samples, 
                        loss_components, component_counts):
        """Compute comprehensive evaluation metrics."""
        metrics = {}
        
        # Loss metrics
        metrics['val_loss'] = total_loss / total_samples if total_samples > 0 else float('inf')
        
        # Loss components
        if component_counts > 0:
            for key, value in loss_components.items():
                metrics[key] = value / component_counts
        
        # Text generation metrics
        if predictions and targets:
            eval_metrics = self.evaluator.compute_all_metrics(predictions, targets)
            metrics.update(eval_metrics)
            
            # Diversity metrics
            unique_preds = len(set(predictions))
            metrics['diversity_score'] = unique_preds / len(predictions)
            metrics['unique_predictions'] = unique_preds
            metrics['total_predictions'] = len(predictions)
            
            # Check for repetitive generation
            metrics['is_repetitive'] = metrics['diversity_score'] < self.config.get('min_diversity_score', 0.3)
            
            # Store examples
            metrics['predictions'] = predictions[:10]  # Store first 10 for logging
            metrics['targets'] = targets[:10]
        else:
            # Empty metrics
            metrics.update({
                'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
                'rouge_l_f': 0.0, 'diversity_score': 0.0,
                'is_repetitive': True, 'unique_predictions': 0
            })
        
        return metrics

    def _update_diversity_tracking(self, metrics):
        """Update diversity history and adaptive weights."""
        diversity_score = metrics.get('diversity_score', 0.0)
        self.diversity_history.append(diversity_score)
        
        # Keep history bounded
        if len(self.diversity_history) > 20:
            self.diversity_history.pop(0)
        
        # Update adaptive scheduler if enabled
        if self.adaptive_scheduler and len(self.diversity_history) >= 3:
            # Get loss components for adaptation
            loss_components = {k: metrics.get(k, 0.0) for k in 
                             ['loss_ce', 'loss_align', 'loss_bow', 'loss_div', 'loss_var']}
            
            # Update weights based on diversity
            self.adaptive_scheduler.update(loss_components, diversity_score)
            
            # Apply new weights to loss function
            new_weights = self.adaptive_scheduler.get_weights()
            self.composite_loss.w_align = new_weights['align']
            self.composite_loss.w_bow = new_weights['bow']
            self.composite_loss.w_div = new_weights['div']
            self.composite_loss.w_var = new_weights['var']
            
            logger.info(f"Updated loss weights: {new_weights}")

    def _log_training_step(self, outputs, pbar):
        """Log training step metrics."""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Create postfix for progress bar
        postfix = {
            'loss': f"{outputs.loss.item():.4f}",
            'lr': f"{current_lr:.1e}"
        }
        
        # Add loss components
        if hasattr(outputs, 'loss_components'):
            for key, value in outputs.loss_components.items():
                short_key = key.replace('loss_', '')[:3]
                postfix[short_key] = f"{value:.3f}"
        
        pbar.set_postfix(postfix)
        
        # Log to wandb
        log_dict = {
            "train/loss": outputs.loss.item(),
            "train/lr": current_lr,
            "step": self.global_step
        }
        
        if hasattr(outputs, 'loss_components'):
            for key, value in outputs.loss_components.items():
                log_dict[f"train/{key}"] = value
        
        # Log current loss weights
        if self.adaptive_scheduler:
            weights = self.adaptive_scheduler.get_weights()
            for name, weight in weights.items():
                log_dict[f"weights/{name}"] = weight
        
        wandb.log(log_dict)

    def _log_epoch_summary(self, epoch, avg_loss, loss_components):
        """Log epoch summary metrics."""
        log_dict = {
            "train/epoch_loss": avg_loss,
            "epoch": epoch
        }
        
        # Log average loss components
        for key, value in loss_components.items():
            log_dict[f"train/epoch_{key}"] = value
        
        # Log diversity trend
        if len(self.diversity_history) > 1:
            recent_diversity = np.mean(self.diversity_history[-5:])
            log_dict["train/recent_diversity"] = recent_diversity
        
        wandb.log(log_dict)
        
        logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        logger.info(f"Loss components: {loss_components}")

    def save_checkpoint(self, epoch, metrics, save_path):
        """Save model checkpoint with training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'global_step': self.global_step,
            'best_bleu4': self.best_bleu4,
            'best_diversity': self.best_diversity,
            'diversity_history': self.diversity_history,
            'composite_loss_state': self.composite_loss.state_dict(),
        }
        
        if self.adaptive_scheduler:
            checkpoint['adaptive_weights'] = self.adaptive_scheduler.get_weights()
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint and restore training state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.composite_loss.load_state_dict(checkpoint['composite_loss_state'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.best_bleu4 = checkpoint.get('best_bleu4', 0.0)
        self.best_diversity = checkpoint.get('best_diversity', 0.0)
        self.diversity_history = checkpoint.get('diversity_history', [])
        
        if 'adaptive_weights' in checkpoint and self.adaptive_scheduler:
            weights = checkpoint['adaptive_weights']
            self.adaptive_scheduler.weights = weights
            # Apply weights to loss function
            self.composite_loss.w_align = weights['align']
            self.composite_loss.w_bow = weights['bow']
            self.composite_loss.w_div = weights['div']
            self.composite_loss.w_var = weights['var']
        
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        return checkpoint.get('epoch', 0)

    def train(self):
        """
        Main training loop with composite loss and anti-collapse mechanisms.
        
        Returns:
            Best BLEU-4 score achieved during training
        """
        logger.info("="*60)
        logger.info("Starting training with composite loss")
        logger.info(f"Initial loss weights: {self.config['loss_weights']}")
        logger.info(f"Adaptive scheduling: {self.config.get('use_adaptive_loss', True)}")
        logger.info("="*60)
        
        consecutive_repetitive = 0
        max_repetitive_tolerance = 5
        
        for epoch in range(self.config['epochs']):
            # Training epoch
            train_loss = self.train_epoch(epoch)
            
            # Log epoch metrics
            wandb.log({
                "train/epoch_loss": train_loss,
                "epoch": epoch
            })
            
            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                val_metrics = self.evaluate()
                
                # Extract key metrics
                bleu4 = val_metrics['bleu_4']
                diversity = val_metrics['diversity_score']
                is_repetitive = val_metrics['is_repetitive']
                
                # Log validation metrics
                self._log_validation_metrics(val_metrics, epoch)
                
                # Check for improvement
                improved = self._check_improvement(bleu4, diversity, is_repetitive)
                
                if improved:
                    # Save best model
                    save_path = os.path.join(self.config['save_dir'], "best_model.pth")
                    self.save_checkpoint(epoch, val_metrics, save_path)
                    logger.info(f"New best model saved - BLEU-4: {bleu4:.3f}, Diversity: {diversity:.3f}")
                    self.patience_counter = 0
                    consecutive_repetitive = 0
                else:
                    self.patience_counter += 1
                    if is_repetitive:
                        consecutive_repetitive += 1
                
                # Handle repetitive generation
                if consecutive_repetitive >= max_repetitive_tolerance:
                    logger.warning(f"Model stuck in repetitive generation for {consecutive_repetitive} evaluations")
                    logger.warning("Consider adjusting loss weights or learning rates")
                
                # Early stopping
                if self.patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Regular checkpointing
            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                save_path = os.path.join(self.config['save_dir'], f"checkpoint_epoch_{epoch+1}.pth")
                self.save_checkpoint(epoch, {}, save_path)
        
        logger.info("="*60)
        logger.info(f"Training completed. Best BLEU-4: {self.best_bleu4:.3f}")
        logger.info(f"Best diversity score: {self.best_diversity:.3f}")
        logger.info("="*60)
        
        return self.best_bleu4

    def _check_improvement(self, bleu4, diversity, is_repetitive):
        """Check if model has improved."""
        # Don't consider repetitive models as improvements
        if is_repetitive:
            return False
        
        # Check for BLEU improvement with diversity constraint
        if bleu4 > self.best_bleu4 and diversity >= self.config.get('min_diversity_score', 0.3):
            self.best_bleu4 = bleu4
            self.best_diversity = max(self.best_diversity, diversity)
            return True
        
        # Also save if diversity significantly improved with reasonable BLEU
        if diversity > self.best_diversity + 0.1 and bleu4 > self.best_bleu4 * 0.9:
            self.best_diversity = diversity
            return True
        
        return False

    def _log_validation_metrics(self, metrics, epoch):
        """Log comprehensive validation metrics."""
        log_dict = {
            "val/loss": metrics['val_loss'],
            "val/bleu_1": metrics['bleu_1'],
            "val/bleu_2": metrics['bleu_2'],
            "val/bleu_3": metrics['bleu_3'],
            "val/bleu_4": metrics['bleu_4'],
            "val/rouge_l": metrics['rouge_l_f'],
            "val/diversity_score": metrics['diversity_score'],
            "val/unique_predictions": metrics['unique_predictions'],
            "val/is_repetitive": float(metrics['is_repetitive']),
            "epoch": epoch
        }
        
        # Log loss components
        for key in ['loss_ce', 'loss_align', 'loss_bow', 'loss_div', 'loss_var']:
            if key in metrics:
                log_dict[f"val/{key}"] = metrics[key]
        
        # Log example predictions
        if 'predictions' in metrics and 'targets' in metrics:
            examples = []
            for pred, target in zip(metrics['predictions'][:5], metrics['targets'][:5]):
                examples.append(f"Target: {target} | Pred: {pred}")
            log_dict["val/examples"] = wandb.Table(
                columns=["Examples"],
                data=[[ex] for ex in examples]
            )
        
        wandb.log(log_dict)
        
        # Console logging
        logger.info(f"Validation - BLEU-4: {metrics['bleu_4']:.3f}, "
                   f"Diversity: {metrics['diversity_score']:.3f}, "
                   f"Loss: {metrics['val_loss']:.4f}")