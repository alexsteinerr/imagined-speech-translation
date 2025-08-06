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
    
    # Model architecture options - 
    'disable_cross_region_attn': False,
    'uniform_region_weight': False,
    'cnn_only': False,
    'disable_cross_modal': False,
    
    # Training parameters
    'epochs': 100,
    'batch_size': 4,  
    'accumulation_steps': 8, 
    'patience': 20,
    'grad_clip_norm': 1.0, 
    
    # Learning rates 
    'brain_encoder_lr': 5e-4,   
    'bart_decoder_lr': 5e-5,     
    'bart_encoder_lr': 1e-5,  
    'projection_lr': 1e-3,    
    'warmup_steps': 1000,
    'weight_decay': 0.01,
    'label_smoothing': 0.01,  

    # Generation parameters
    'num_beams': 3,             
    'max_gen_length': 18,      
    'no_repeat_ngram_size': 2,
    'length_penalty': 0.8,    
    'min_length': 4,             
    'do_sample': False,     
    'temperature': 0.7,   
    'top_k': 40,           
    'top_p': 0.85,          
    
    # Data splits
    'train_split': 0.8,
    'val_split': 0.1,
    'val_max_samples': 2000,
    
    # Experiment settings
    'experiment_name': 'EEG-Chinese-Fixed',  
    'log_interval': 25,        
    'eval_interval': 1,       
    'save_every_n_epochs': 3,    
    
    # Advanced training options
    'use_amp': False,
    'find_unused_parameters': True,
    'dataloader_num_workers': 0,
    'pin_memory': True,
    'persistent_workers': False,
    
    # Regularization
    'dropout_rate': 0.08,     
    'attention_dropout': 0.05,  
    'classifier_dropout': 0.1,
    
    # Data augmentation 
    'data_augmentation': True,
    'augment_prob': 0.1,       
    'noise_std_ratio': 0.008,   
    'scale_range': (0.99, 1.01), 
    
    # Validation and testing
    'val_check_interval': 0.5,
    'test_split': 0.1,
    'early_stopping_metric': 'bleu_4',
    'early_stopping_mode': 'max',
    
    # Logging and monitoring
    'log_every_n_steps': 10,
    'save_top_k': 3,
    'monitor_metric': 'val_bleu_4',
    'log_predictions': True,
    'max_prediction_examples': 8,  
    
    # Model specific parameters
    'freeze_bart_encoder': False,
    'freeze_bart_embeddings': False, 
    'region_attention_heads': 8,
    'region_fusion_layers': 1,
    
    # EEG-SPECIFIC PARAMETERS
    'eeg_conditioning_strength_init': 0.7, 
    'eeg_conditioning_clamp_min': 0.3,  
    'eeg_conditioning_clamp_max': 1.2,      
    'encoder_sequence_length': 24,        
    'eeg_projection_layers': 3,        
    
    # Reproducibility
    'seed': 42,
    'deterministic': True,
    'benchmark': False,
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