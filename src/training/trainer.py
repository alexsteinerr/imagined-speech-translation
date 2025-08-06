"""
Training loop and utilities for EEG-to-text model.
"""

import os
import time
import torch
import wandb
import numpy as np
from tqdm.auto import tqdm
from ..evaluation.evaluator import ChineseEvaluator
import logging

logger = logging.getLogger(__name__)


class EEGTrainer:
    """
    Trainer class for EEG-to-text model.
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

    def safe_forward_pass(self, eeg, decoder_input_ids, labels):
        """Safe forward pass with error handling."""
        try:
            vocab_size = len(self.tokenizer.get_vocab())
            
            # Validate token IDs
            if decoder_input_ids.max() >= vocab_size:
                return None
            if (labels != -100).any() and labels[labels != -100].max() >= vocab_size:
                return None
            
            outputs = self.model(
                eeg_data=eeg,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                label_smoothing_factor=self.config['label_smoothing']
            )
            
            return outputs
            
        except RuntimeError as e:
            if "CUDA error" in str(e) or "out of memory" in str(e):
                torch.cuda.empty_cache()
                return None
            else:
                raise e

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        accumulation_step = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            try:
                eeg = [region.to(self.device) for region in batch['eeg']]
                decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.safe_forward_pass(eeg, decoder_input_ids, labels)
                
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
                
                # Logging
                if step % self.config['log_interval'] == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'loss': f"{outputs.loss.item():.4f}",
                        'lr': f"{current_lr:.1e}"
                    })
                    
                    wandb.log({
                        "train/step_loss": outputs.loss.item(),
                        "train/lr": current_lr,
                        "step": self.global_step
                    })
                    
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
        return avg_loss

    def evaluate(self):
        """Evaluate model on validation set."""
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
                    
                    # Calculate validation loss
                    val_outputs = self.safe_forward_pass(eeg, decoder_input_ids, labels)
                    if val_outputs is not None and val_outputs.loss is not None:
                        total_val_loss += val_outputs.loss.item() * len(labels)
                        total_val_samples += len(labels)
                    
                    # Generate predictions
                    gen_config = {
                        'max_length': self.config['max_gen_length'],
                        'num_beams': self.config['num_beams'],
                        'length_penalty': self.config['length_penalty'],
                        'no_repeat_ngram_size': self.config['no_repeat_ngram_size'],
                        'min_length': self.config['min_length'],
                        'do_sample': False,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'eos_token_id': self.tokenizer.eos_token_id
                    }
                    
                    if self.config['num_beams'] > 1:
                        gen_config['early_stopping'] = True
                    
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
            return {
                'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 
                'rouge_l_f': 0.0, 'val_loss': avg_val_loss,
                'predictions': [], 'targets': []
            }
        
        try:
            metrics = self.evaluator.compute_all_metrics(all_predictions, all_targets)
            metrics['val_loss'] = avg_val_loss
            metrics['predictions'] = all_predictions
            metrics['targets'] = all_targets
        except:
            metrics = {
                'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 
                'rouge_l_f': 0.0, 'val_loss': avg_val_loss,
                'predictions': all_predictions, 'targets': all_targets
            }
        
        return metrics

    def log_predictions_to_wandb(self, predictions, targets, epoch, max_samples=10):
        """Log sample predictions and targets to wandb as a table."""
        # Limit the number of samples to avoid overwhelming wandb
        num_samples = min(len(predictions), max_samples)
        
        # Create data for wandb table
        table_data = []
        for i in range(num_samples):
            table_data.append([
                i + 1,
                targets[i],
                predictions[i],
                len(targets[i].split()),  # target length
                len(predictions[i].split())  # prediction length
            ])
        
        # Create wandb table
        table = wandb.Table(
            columns=["Sample", "Target", "Prediction", "Target_Length", "Pred_Length"],
            data=table_data
        )
        
        wandb.log({
            f"predictions_table_epoch_{epoch}": table,
            "epoch": epoch
        })

    def log_region_weights_to_wandb(self, prefix="", step=None):
        """Log brain region weights to wandb using model.brain_encoder.get_region_weights()."""
        if not hasattr(self.model, 'brain_encoder'):
            return
        
        try:
            # Get region weights from the model
            weights_info = self.model.brain_encoder.get_region_weights()
            log_dict = {}
            
            # Log individual region weights
            for name, weight in zip(weights_info['names'], weights_info['softmax']):
                log_dict[f"{prefix}region_weights/{name}"] = float(weight)
            
            # Log raw importance parameters if available
            if hasattr(self.model.brain_encoder, 'region_importance') and not self.model.brain_encoder.uniform_region_weight:
                raw_weights = self.model.brain_encoder.region_importance.data.cpu().numpy()
                for name, raw_weight in zip(weights_info['names'], raw_weights):
                    log_dict[f"{prefix}region_importance_raw/{name}"] = float(raw_weight)
            
            # Add step if provided
            if step is not None:
                log_dict["step"] = step
            
            wandb.log(log_dict)
            
        except Exception as e:
            logger.warning(f"Failed to log region weights: {e}")

    def log_region_weights_table_to_wandb(self, prefix="", step=None):
        """Log region weights as a table to wandb for better visualization."""
        if not hasattr(self.model, 'brain_encoder'):
            return
        
        try:
            # Get region weights from the model
            weights_info = self.model.brain_encoder.get_region_weights()
            
            # Prepare data for the table
            table_data = []
            for name, weight in zip(weights_info['names'], weights_info['softmax']):
                table_data.append([name, float(weight)])
            
            # Create wandb table
            table = wandb.Table(
                columns=["Region", "Weight"],
                data=table_data
            )
            
            log_dict = {f"{prefix}region_weights_table": table}
            if step is not None:
                log_dict["step"] = step
            
            wandb.log(log_dict)
            
        except Exception as e:
            logger.warning(f"Failed to log region weights table: {e}")

    def save_checkpoint(self, epoch, metrics, save_path):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'global_step': self.global_step
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint: {save_path}")

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.config['epochs']} epochs...")
        
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
                val_loss = val_metrics['val_loss']
                predictions = val_metrics['predictions']
                targets = val_metrics['targets']
                
                # Log metrics to wandb
                wandb.log({
                    "val/loss": val_loss,
                    "val/bleu_1": val_metrics['bleu_1'],
                    "val/bleu_2": val_metrics['bleu_2'],
                    "val/bleu_3": val_metrics['bleu_3'],
                    "val/bleu_4": bleu4,
                    "val/rouge_l": val_metrics['rouge_l_f'],
                    "epoch": epoch
                })
                

                self.log_region_weights_to_wandb(prefix="val/", step=self.global_step)
            
                if epoch % 5 == 0:  
                    self.log_region_weights_table_to_wandb(
                        prefix=f"val/epoch_{epoch}_", step=self.global_step
                    )
                

                if predictions and targets:
                    self.log_predictions_to_wandb(predictions, targets, epoch)
                
                # Save best model
                if bleu4 > self.best_bleu4:
                    self.best_bleu4 = bleu4
                    save_path = os.path.join(self.config['save_dir'], "best_model.pth")
                    # Remove predictions and targets from checkpoint to save space
                    checkpoint_metrics = {k: v for k, v in val_metrics.items() 
                                        if k not in ['predictions', 'targets']}
                    self.save_checkpoint(epoch, checkpoint_metrics, save_path)
                    logger.info(f"New best BLEU-4: {bleu4:.3f}")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Regular checkpoints
            if (epoch + 1) % self.config['save_every_n_epochs'] == 0:
                save_path = os.path.join(self.config['save_dir'], f"checkpoint_epoch_{epoch+1}.pth")
                self.save_checkpoint(epoch, {}, save_path)
        
        logger.info(f"Training completed. Best BLEU-4: {self.best_bleu4:.3f}")
        return self.best_bleu4