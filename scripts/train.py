#!/usr/bin/env python3
"""
Main training script for EEG-to-text model with composite loss.
"""

import os
import sys
import torch
import torch.nn as nn
import wandb
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.eeg_model import EEGDecodingModel
from src.data.dataset import EEGDataset
from src.training.trainer import EEGTrainer
from config.training_config import CONFIG, get_optimizer_groups, validate_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make training deterministic if specified
    if CONFIG.get('deterministic', True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seeds set to {seed}")


def setup_model_and_tokenizer(device):
    """Setup model and tokenizer."""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['pretrained_model'])
    
    # Ensure tokenizer has necessary tokens
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    vocab_size = len(tokenizer.get_vocab())
    logger.info(f"Tokenizer loaded. Vocab size: {vocab_size}")
    
    # Log special tokens for debugging
    logger.info(f"Tokenizer special tokens: "
                f"PAD={tokenizer.pad_token_id}, "
                f"EOS={tokenizer.eos_token_id}, "
                f"BOS={tokenizer.bos_token_id}")
    
    # Get region channel counts (will be set from dataset)
    region_channel_counts = {
        'frontal': 16,   # Will be updated from dataset
        'temporal': 9,
        'central': 11,
        'parietal': 12
    }
    
    logger.info("Creating model...")
    model = EEGDecodingModel(
        n_timepoints=CONFIG['n_timepoints'],
        region_channel_counts=region_channel_counts,
        hidden_dim=CONFIG['hidden_dim'],
        disable_cross_region_attn=CONFIG.get('disable_cross_region_attn', False),
        uniform_region_weight=CONFIG.get('uniform_region_weight', False),
        cnn_only=CONFIG.get('cnn_only', False)
    )
    
    # Move model to device
    model = model.to(device)
    
    # CRITICAL FIX: Validate and resize embeddings if needed
    model_vocab_size = model.bart_decoder.bart.get_input_embeddings().weight.size(0)
    if vocab_size != model_vocab_size:
        logger.warning(f"Vocab size mismatch! Tokenizer: {vocab_size}, Model: {model_vocab_size}")
        logger.warning("Resizing model embeddings to match tokenizer")
        model.bart_decoder.bart.resize_token_embeddings(vocab_size)
    
    # Initialize weights for non-pretrained components
    initialize_custom_weights(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created. Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    return model, tokenizer


def initialize_custom_weights(model):
    """Initialize weights for custom (non-pretrained) components."""
    for name, param in model.named_parameters():
        # Skip BART pretrained weights
        if 'bart' in name.lower():
            continue
            
        if 'weight' in name:
            if 'norm' in name or 'layer_norm' in name:
                torch.nn.init.ones_(param)
            elif 'embedding' in name:
                torch.nn.init.normal_(param, std=0.02)
            elif len(param.shape) >= 2:
                # Use smaller initialization for stability
                torch.nn.init.xavier_uniform_(param, gain=0.02)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)
    
    logger.info("Custom weights initialized")


def setup_data_loaders(tokenizer):
    """Setup data loaders with proper splits."""
    logger.info("Loading dataset...")
    
    # Load full dataset
    dataset = EEGDataset(
        data_dir=CONFIG['data_dir'],
        csv_path=CONFIG['montage_file'],
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
        data_augmentation=CONFIG['augmentation']['enabled']
    )
    
    # Get dataset statistics
    stats = dataset.get_sample_stats()
    logger.info(f"Dataset loaded: {stats['total_samples']} samples")
    logger.info(f"Region channel counts: {stats['region_channel_counts']}")
    
    # Calculate split sizes
    n = len(dataset)
    train_n = int(CONFIG['train_split'] * n)
    val_n = int(CONFIG['val_split'] * n)
    test_n = n - (train_n + val_n)
    
    # Create splits with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(CONFIG['seed'])
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n],
        generator=generator
    )
    
    logger.info(f"Data splits - Train: {train_n}, Val: {val_n}, Test: {test_n}")
    
    # IMPORTANT: Disable pin_memory on Windows to avoid CUDA errors
    use_pin_memory = torch.cuda.is_available() and sys.platform != 'win32'
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,  # Always use 0 on Windows
        pin_memory=use_pin_memory,  # Disabled on Windows
        drop_last=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    
    logger.info(f"DataLoader settings - pin_memory: {use_pin_memory}, num_workers: 0")
    
    return train_loader, val_loader, test_loader, stats['region_channel_counts']


def setup_optimizer_and_scheduler(model, train_loader):
    """Setup optimizer with parameter groups and scheduler."""
    # Get parameter groups with different learning rates
    param_groups = get_optimizer_groups(model)
    
    # Log parameter groups
    for i, group in enumerate(param_groups):
        n_params = sum(p.numel() for p in group['params'])
        logger.info(f"Param group {i}: {n_params:,} params, lr={group['lr']:.2e}")
    
    # Create optimizer
    optimizer = AdamW(
        param_groups,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=CONFIG['weight_decay']
    )
    
    # Calculate total training steps
    steps_per_epoch = len(train_loader) // CONFIG['accumulation_steps']
    total_steps = steps_per_epoch * CONFIG['epochs']
    warmup_steps = CONFIG.get('warmup_steps', 500)
    
    logger.info(f"Training steps: {total_steps} total, {warmup_steps} warmup")
    
    # Create scheduler (cosine is often better than linear)
    scheduler_type = CONFIG.get('scheduler_type', 'cosine')
    if scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    logger.info(f"Optimizer and {scheduler_type} scheduler created")
    
    return optimizer, scheduler


def update_model_with_dataset_info(model, region_channel_counts):
    """Update model's region channel counts from dataset."""
    # Reinitialize brain encoder with correct channel counts if needed
    if hasattr(model, 'brain_encoder'):
        current_counts = model.brain_encoder.region_channel_counts
        if current_counts != region_channel_counts:
            logger.warning("Region channel counts mismatch. Reinitializing brain encoder...")
            model.brain_encoder = model.brain_encoder.__class__(
                n_timepoints=CONFIG['n_timepoints'],
                region_channel_counts=region_channel_counts,
                hidden_dim=CONFIG['hidden_dim'],
                disable_cross_region_attn=CONFIG.get('disable_cross_region_attn', False),
                uniform_region_weight=CONFIG.get('uniform_region_weight', False),
                cnn_only=CONFIG.get('cnn_only', False)
            ).to(next(model.parameters()).device)
            logger.info("Brain encoder reinitialized with correct channel counts")


def setup_wandb():
    """Setup Weights & Biases logging."""
    # Create run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = CONFIG.get('experiment_name', f'eeg_training_{timestamp}')
    
    # Initialize wandb
    wandb.init(
        project=CONFIG.get('wandb_project', 'EEG-Chinese'),
        name=run_name,
        config=CONFIG,
        tags=['composite_loss', 'anti_collapse'],
        notes=f"Training with composite loss and adaptive scheduling"
    )
    
    logger.info(f"W&B initialized: {run_name}")


def main():
    """Main training function."""
    # Validate configuration
    validate_config()
    
    # Set random seeds
    set_random_seeds(CONFIG['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

        # Set for better error messages
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        # Disable cudnn for debugging
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # Setup wandb
    setup_wandb()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(device)
    
    # Setup data loaders
    train_loader, val_loader, test_loader, region_channel_counts = setup_data_loaders(tokenizer)
    
    # Update model with actual channel counts from dataset
    update_model_with_dataset_info(model, region_channel_counts)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, train_loader)
    
    # Create save directory
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {CONFIG['save_dir']}")
    
    # Create trainer with composite loss
    trainer = EEGTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=CONFIG
    )
    
    # Start training
    logger.info("="*60)
    logger.info("Starting training with composite loss")
    logger.info(f"Epochs: {CONFIG['epochs']}")
    logger.info(f"Batch size: {CONFIG['batch_size']}")
    logger.info(f"Accumulation steps: {CONFIG['accumulation_steps']}")
    logger.info(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['accumulation_steps']}")
    logger.info("="*60)
    
    try:
        best_score = trainer.train()
        
        logger.info("="*60)
        logger.info(f"Training completed successfully!")
        logger.info(f"Best BLEU-4: {best_score:.3f}")
        logger.info(f"Best model saved at: {os.path.join(CONFIG['save_dir'], 'best_model.pth')}")
        logger.info("="*60)
        
        # Final evaluation on test set
        if test_loader is not None:
            logger.info("Running final evaluation on test set...")
            trainer.val_loader = test_loader  # Temporarily replace val loader
            test_metrics = trainer.evaluate()
            logger.info(f"Test BLEU-4: {test_metrics['bleu_4']:.3f}")
            logger.info(f"Test Diversity: {test_metrics['diversity_score']:.3f}")
            
            # Log test metrics to wandb
            wandb.log({
                "test/bleu_4": test_metrics['bleu_4'],
                "test/diversity_score": test_metrics['diversity_score'],
                "test/loss": test_metrics['val_loss']
            })
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info(f"Latest checkpoint saved at: {os.path.join(CONFIG['save_dir'], 'interrupted_checkpoint.pth')}")
        trainer.save_checkpoint(trainer.epoch, {}, os.path.join(CONFIG['save_dir'], 'interrupted_checkpoint.pth'))
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        # Cleanup
        wandb.finish()
        torch.cuda.empty_cache()
        logger.info("Training script finished")


if __name__ == "__main__":
    main()