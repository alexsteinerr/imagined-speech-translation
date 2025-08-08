"""
Optimized training configuration for EEG-to-text model with composite loss.
"""

CONFIG = {
    'data_dir': 'data/eeg_data/',
    'montage_file': 'data/montage.csv',
    'save_dir': './checkpoints/',
    
    'pretrained_model': 'fnlp/bart-base-chinese',
    'hidden_dim': 768,
    'n_timepoints': 1651,
    'max_length': 16,
    
    # Model architecture flags
    'disable_cross_region_attn': False, 
    'uniform_region_weight': False,  
    'cnn_only': False,             
    
    'epochs': 100,
    'batch_size': 4,
    'accumulation_steps': 8,  # Effective batch size = 32
    'patience': 15,          
    'grad_clip_norm': 1.0,
    
    'brain_encoder_lr': 3e-4,    
    'bart_decoder_lr': 3e-5,    
    'projection_lr': 5e-4,    
    'warmup_steps': 500,      
    'weight_decay': 0.01,
    
    'loss_weights': {
        'ce': 1.0,          
        'align': 0.5,        
        'bow': 0.15,     
        'div': 0.1,      
        'var': 0.05,         
    },
    
    'use_adaptive_loss': True,
    'adaptation_rate': 0.01,
    'adaptation_patience': 5,  

    'label_smoothing': 0.05,  
    'bow_vocab_size': 2000, 
    'contrastive_tau': 0.07,   

    'generation': {
        'eval': {
            'num_beams': 5,         
            'max_length': 18,
            'min_length': 4,
            'no_repeat_ngram_size': 3,
            'repetition_penalty': 1.3,  
            'length_penalty': 1.0,   
            'diversity_penalty': 0.5,    
            'num_beam_groups': 2,       
            'early_stopping': True,
        },
        'train': {
            'do_sample': True,
            'max_length': 18,
            'min_length': 4,
            'temperature': 0.8,
            'top_p': 0.9,             
            'top_k': 50,
            'repetition_penalty': 1.2,
            'no_repeat_ngram_size': 2,
        }
    },
    
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'batch_size': 4,
    'num_workers': 0,         

    'augmentation': {
        'enabled': True,
        'noise_std': 0.01, 
        'scale_range': (0.95, 1.05), 
        'time_shift': 2,     
    },
    
    'experiment_name': 'EEG-Chinese-CompositeLoss',
    'log_interval': 20,      
    'eval_interval': 1,      
    'save_interval': 5,       
    
    # Metrics to track
    'monitor_metrics': {
        'primary': 'val_bleu_4',          
        'secondary': 'val_diversity_score',
        'early_stopping': 'val_bleu_4',
        'mode': 'max',
    },
    
    'seed': 42,
    'deterministic': True,
    'mixed_precision': False, 
    
    # Thresholds and limits
    'min_diversity_score': 0.3,
    'max_diversity_score': 0.8,  
    'max_gradient_norm': 1.0,
    
}

def get_optimizer_groups(model):
    """
    Create parameter groups with different learning rates.
    More granular than before for better control.
    """
    groups = {
        'brain_encoder': [],
        'projection': [],
        'bart_decoder': [],
        'loss_heads': [],  # BoW and alignment projections
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'brain_encoder' in name:
            groups['brain_encoder'].append(param)
        elif 'eeg_to_bart' in name or 'projection' in name:
            groups['projection'].append(param)
        elif 'bow_head' in name or '_proj' in name:
            groups['loss_heads'].append(param)
        elif 'bart' in name:
            groups['bart_decoder'].append(param)
        else:
            groups['projection'].append(param)  # Default group
    
    param_groups = [
        {'params': groups['brain_encoder'], 'lr': CONFIG['brain_encoder_lr']},
        {'params': groups['projection'], 'lr': CONFIG['projection_lr']},
        {'params': groups['bart_decoder'], 'lr': CONFIG['bart_decoder_lr']},
        {'params': groups['loss_heads'], 'lr': CONFIG['projection_lr'] * 0.5},
    ]
    
    # Filter out empty groups
    param_groups = [g for g in param_groups if g['params']]
    
    # Add weight decay
    for group in param_groups:
        group['weight_decay'] = CONFIG['weight_decay']
    
    return param_groups


def get_loss_config():
    """Get composite loss configuration."""
    return {
        'vocab_size': None,
        'hidden_dim': CONFIG['hidden_dim'],
        'pad_token_id': None, 
        'bow_indices': None,
        'label_smoothing': CONFIG['label_smoothing'],
        'w_ce': CONFIG['loss_weights']['ce'],
        'w_align': CONFIG['loss_weights']['align'],
        'w_bow': CONFIG['loss_weights']['bow'],
        'w_div': CONFIG['loss_weights']['div'],
        'w_var': CONFIG['loss_weights']['var'],
        'tau': CONFIG['contrastive_tau'],
    }


def get_scheduler_config(num_training_steps):
    """Get learning rate scheduler configuration."""
    return {
        'scheduler_type': 'cosine_with_warmup', 
        'num_warmup_steps': CONFIG['warmup_steps'],
        'num_training_steps': num_training_steps,
        'min_lr_ratio': 0.1
    }


def should_trigger_adaptation(metrics):
    """
    Determine if loss weight adaptation should be triggered.
    
    Args:
        metrics: Dictionary of current metrics
        
    Returns:
        str: Adaptation action ('increase_div', 'decrease_div', or None)
    """
    diversity = metrics.get('diversity_score', 0.5)
    
    if diversity < CONFIG['min_diversity_score']:
        return 'increase_diversity'
    elif diversity > CONFIG['max_diversity_score']:
        return 'decrease_diversity'
    
    # Check for repetitive generation
    if metrics.get('unique_ratio', 1.0) < 0.5:
        return 'increase_diversity'
    
    return None


def update_loss_weights(current_weights, action, rate=None):
    """
    Update loss weights based on adaptation action.
    
    Args:
        current_weights: Current loss weight dictionary
        action: Adaptation action string
        rate: Adaptation rate (uses CONFIG default if None)
        
    Returns:
        Updated weights dictionary
    """
    rate = rate or CONFIG['adaptation_rate']
    weights = current_weights.copy()
    
    if action == 'increase_diversity':
        weights['div'] = min(0.3, weights['div'] * (1 + rate))
        weights['var'] = min(0.15, weights['var'] * (1 + rate))
        weights['align'] = min(0.7, weights['align'] * (1 + rate * 0.5))
        
    elif action == 'decrease_diversity':
        weights['div'] = max(0.05, weights['div'] * (1 - rate))
        weights['var'] = max(0.02, weights['var'] * (1 - rate))
        
    return weights

def validate_config():
    """Validate configuration consistency."""
    assert CONFIG['train_split'] + CONFIG['val_split'] + CONFIG['test_split'] == 1.0
    assert CONFIG['batch_size'] > 0
    assert CONFIG['accumulation_steps'] > 0
    assert all(w > 0 for w in CONFIG['loss_weights'].values())
    assert CONFIG['min_diversity_score'] < CONFIG['max_diversity_score']
    return True