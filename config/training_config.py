"""
Training configuration for EEG-to-text model.
"""

CONFIG = {
    # Data and paths
    'data_dir': 'data/eeg_data/',
    'montage_file': 'data/montage.csv',
    'save_dir': './checkpoints/',
    
    # Model configuration
    'pretrained_model': 'fnlp/bart-base-chinese',
    'hidden_dim': 768,
    'n_timepoints': 1651,
    'max_length': 16,
    
    # Model architecture options
    'disable_cross_region_attn': False,
    'uniform_region_weight': False,
    'cnn_only': False,
    'disable_cross_modal': False,
    
    # Training parameters
    'epochs': 100,
    'batch_size': 4,
    'accumulation_steps': 8,
    'patience': 20,
    'grad_clip_norm': 5.0,
    
    # Learning rates
    'brain_encoder_lr': 1e-4,
    'bart_decoder_lr': 1e-5,
    'bart_encoder_lr': 5e-6,
    'projection_lr': 5e-4,
    'warmup_steps': 1000,
    'weight_decay': 0.01,
    'label_smoothing': 0.05,
    
    # Generation parameters
    'num_beams': 4,
    'max_gen_length': 20,
    'no_repeat_ngram_size': 2,
    'length_penalty': 1.0,
    'min_length': 3,
    'do_sample': True,
    'temperature': 0.8,
    'top_k': 50,
    'top_p': 0.9,
    
    # Data splits
    'train_split': 0.8,
    'val_split': 0.1,
    'val_max_samples': 2000,
    
    # Experiment settings
    'experiment_name': 'EEG-Chinese',
    'log_interval': 50,
    'eval_interval': 2,
    'save_every_n_epochs': 5,
    
    # Advanced training options
    'use_amp': False,                    # Automatic mixed precision
    'find_unused_parameters': True,      # For DDP training
    'dataloader_num_workers': 0,         # Number of data loading workers
    'pin_memory': True,                  # Pin memory for faster GPU transfer
    'persistent_workers': False,         # Keep workers alive between epochs
    
    # Regularization
    'dropout_rate': 0.1,                 # General dropout rate
    'attention_dropout': 0.1,            # Attention dropout rate
    'classifier_dropout': 0.1,           # Classifier dropout rate
    
    # Data augmentation
    'data_augmentation': True,           # Enable EEG data augmentation
    'augment_prob': 0.15,               # Probability of applying augmentation
    'noise_std_ratio': 0.01,            # Noise standard deviation ratio
    'scale_range': (0.98, 1.02),        # Scaling range for augmentation
    
    # Validation and testing
    'val_check_interval': 0.5,          # Validation check interval (fraction of epoch)
    'test_split': 0.1,                  # Test set split ratio
    'early_stopping_metric': 'bleu_4',  # Metric for early stopping
    'early_stopping_mode': 'max',       # 'max' or 'min' for early stopping
    
    # Logging and monitoring
    'log_every_n_steps': 10,            # Log training metrics every n steps
    'save_top_k': 3,                    # Save top k checkpoints
    'monitor_metric': 'val_bleu_4',     # Metric to monitor for checkpointing
    'log_predictions': True,            # Log prediction examples
    'max_prediction_examples': 5,       # Max prediction examples to log
    
    # Model specific parameters
    'freeze_bart_encoder': False,       # Freeze BART encoder during training
    'freeze_bart_embeddings': True,     # Freeze BART embeddings
    'region_attention_heads': 8,        # Number of attention heads for region fusion
    'region_fusion_layers': 1,          # Number of transformer layers for region fusion
    
    # Generation evaluation parameters
    'eval_generation_config': {
        'num_beams': 4,
        'max_length': 20,
        'min_length': 3,
        'length_penalty': 1.0,
        'no_repeat_ngram_size': 2,
        'do_sample': False,
        'early_stopping': True
    },
    
    
    # Reproducibility
    'seed': 42,                        # Random seed
    'deterministic': True,             # Use deterministic algorithms
    'benchmark': False,                # Enable cudnn benchmark
}

def get_optimizer_config():
    """Get optimizer-specific configuration."""
    return {
        'brain_encoder_lr': CONFIG['brain_encoder_lr'],
        'bart_decoder_lr': CONFIG['bart_decoder_lr'],
        'bart_encoder_lr': CONFIG['bart_encoder_lr'],
        'projection_lr': CONFIG['projection_lr'],
        'warmup_steps': CONFIG['warmup_steps'],
        'weight_decay': CONFIG['weight_decay'],
    }


def get_generation_config():
    """Get text generation configuration."""
    return {
        'num_beams': CONFIG['num_beams'],
        'max_gen_length': CONFIG['max_gen_length'],
        'no_repeat_ngram_size': CONFIG['no_repeat_ngram_size'],
        'length_penalty': CONFIG['length_penalty'],
        'min_length': CONFIG['min_length'],
        'do_sample': CONFIG['do_sample'],
        'temperature': CONFIG['temperature'],
        'top_k': CONFIG['top_k'],
        'top_p': CONFIG['top_p'],
    }


def get_data_config():
    """Get data processing configuration."""
    return {
        'data_dir': CONFIG['data_dir'],
        'montage_file': CONFIG['montage_file'],
        'max_length': CONFIG['max_length'],
        'train_split': CONFIG['train_split'],
        'val_split': CONFIG['val_split'],
        'test_split': CONFIG['test_split'],
        'batch_size': CONFIG['batch_size'],
        'dataloader_num_workers': CONFIG['dataloader_num_workers'],
        'pin_memory': CONFIG['pin_memory'],
        'persistent_workers': CONFIG['persistent_workers'],
        'data_augmentation': CONFIG['data_augmentation'],
        'augment_prob': CONFIG['augment_prob'],
        'noise_std_ratio': CONFIG['noise_std_ratio'],
        'scale_range': CONFIG['scale_range'],
    }


def get_training_config():
    """Get training-specific configuration."""
    return {
        'epochs': CONFIG['epochs'],
        'accumulation_steps': CONFIG['accumulation_steps'],
        'grad_clip_norm': CONFIG['grad_clip_norm'],
        'patience': CONFIG['patience'],
        'label_smoothing': CONFIG['label_smoothing'],
        'use_amp': CONFIG['use_amp'],
        'gradient_checkpointing': CONFIG['gradient_checkpointing'],
        'freeze_bart_encoder': CONFIG['freeze_bart_encoder'],
        'freeze_bart_embeddings': CONFIG['freeze_bart_embeddings'],
    }