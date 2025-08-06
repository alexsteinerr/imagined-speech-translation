#!/usr/bin/env python3
"""
Main training script for EEG-to-text model.
"""

import os
import sys
import torch
import wandb
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.eeg_model import EEGDecodingModel
from src.data.dataset import EEGDataset
from src.training.trainer import EEGTrainer
from config.training_config import CONFIG, get_optimizer_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model_and_data():
    """Setup model, tokenizer, and data loaders."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['pretrained_model'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load dataset
    dataset = EEGDataset(
        CONFIG['data_dir'],
        CONFIG['montage_file'],
        tokenizer,
        max_length=CONFIG['max_length'],
        data_augmentation=True
    )
    
    # Get dataset statistics
    stats = dataset.get_sample_stats()
    logger.info(f"Dataset loaded: {stats['total_samples']} samples")
    logger.info(f"Region channel counts: {stats['region_channel_counts']}")
    
    # Create model
    model = EEGDecodingModel(
        n_timepoints=CONFIG['n_timepoints'],
        region_channel_counts=stats['region_channel_counts'],
    ).to(device)
    
    # Initialize weights
    initialize_weights(model)
    
    # Split dataset
    n = len(dataset)
    train_n = int(CONFIG['train_split'] * n)
    val_n = int(CONFIG['val_split'] * n)
    test_n = n - (train_n + val_n)
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Data splits - Train: {train_n}, Val: {val_n}, Test: {test_n}")
    
    return model, tokenizer, train_loader, val_loader, test_loader, device


def initialize_weights(model):
    """Initialize model weights."""
    for name, param in model.named_parameters():
        if 'bart' not in name.lower():
            if 'weight' in name:
                if 'norm' in name or 'layer_norm' in name:
                    torch.nn.init.ones_(param)
                elif 'embedding' in name:
                    torch.nn.init.normal_(param, std=0.02)
                elif len(param.shape) >= 2:
                    torch.nn.init.xavier_uniform_(param, gain=0.02)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)


def setup_optimizer_and_scheduler(model, train_loader):
    """Setup optimizer with different learning rates for different components."""
    optimizer_config = get_optimizer_config()
    
    # Separate parameter groups
    brain_encoder_params = []
    bart_encoder_params = []
    bart_decoder_params = []
    projection_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'brain_encoder' in name:
            brain_encoder_params.append(param)
        elif 'eeg_to_bart' in name or 'projection' in name:
            projection_params.append(param)
        elif 'bart' in name and 'encoder' in name:
            bart_encoder_params.append(param)
        elif 'bart' in name:
            bart_decoder_params.append(param)
        else:
            projection_params.append(param)
    
    param_groups = []
    
    if brain_encoder_params:
        param_groups.append({
            'params': brain_encoder_params,
            'lr': optimizer_config['brain_encoder_lr'],
            'weight_decay': optimizer_config['weight_decay']
        })
    
    if projection_params:
        param_groups.append({
            'params': projection_params,
            'lr': optimizer_config['projection_lr'],
            'weight_decay': optimizer_config['weight_decay']
        })
    
    if bart_encoder_params:
        param_groups.append({
            'params': bart_encoder_params,
            'lr': optimizer_config['bart_encoder_lr'],
            'weight_decay': 0.0
        })
    
    if bart_decoder_params:
        param_groups.append({
            'params': bart_decoder_params,
            'lr': optimizer_config['bart_decoder_lr'],
            'weight_decay': 0.0
        })
    
    optimizer = AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))
    
    # Setup scheduler
    total_steps = len(train_loader) * CONFIG['epochs'] // CONFIG['accumulation_steps']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=optimizer_config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler


def main():
    """Main training function."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize wandb
    wandb.init(
        project='EEG-Chinese',
        name=CONFIG['experiment_name'],
        config=CONFIG
    )
    
    # Setup model and data
    model, tokenizer, train_loader, val_loader, test_loader, device = setup_model_and_data()
    optimizer, scheduler = setup_optimizer_and_scheduler(model, train_loader)
    
    # Create save directory
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # Create trainer
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
    try:
        best_score = trainer.train()
        logger.info(f"Training completed successfully. Best BLEU-4: {best_score:.3f}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()