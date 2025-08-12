"""
Balanced recovery configuration for EEG-to-text model.
This configuration gradually breaks collapse without causing gibberish.
"""

CONFIG = {
    # ============= Data and Paths =============
    'data_dir': 'data/eeg_data/',
    'montage_file': 'data/montage.csv',
    'save_dir': './checkpoints/',
    
    # ============= Model Configuration =============
    'pretrained_model': 'fnlp/bart-base-chinese',
    'hidden_dim': 768,
    'n_timepoints': 1651,
    'max_length': 16,
    
    # Model architecture flags
    'disable_cross_region_attn': False, 
    'uniform_region_weight': False,  
    'cnn_only': False,             
    
    # ============= Training Parameters =============
    'epochs': 100,
    'batch_size': 4,
    'accumulation_steps': 8,  # Effective batch size = 32
    'patience': 20,           # More patience during recovery
    'grad_clip_norm': 1.0,    # Standard clipping
    
    # ============= BALANCED Learning Rates =============
    'brain_encoder_lr': 3e-4,    # Normal rate for encoder
    'bart_decoder_lr': 3e-5,      # Conservative for decoder
    'projection_lr': 1e-4,        # Moderate for projections
    'warmup_steps': 500,          # Standard warmup
    'weight_decay': 0.01,         # Standard decay
    
    # ============= STAGED Loss Weights (Start Conservative) =============
    'loss_weights': {
        'ce': 1.0,      # Keep CE as anchor
        'align': 0.3,   # Start low, will increase
        'bow': 0.1,     # Start low
        'div': 0.2,     # Moderate diversity
        'var': 0.05,    # Light variance
    },
    
    # Staged adaptation - gradually increase anti-collapse weights
    'use_adaptive_loss': True,
    'adaptation_rate': 0.02,      # Moderate adaptation
    'adaptation_patience': 3,     # Wait before adapting
    
    # Stage-based weight scheduling
    'staged_loss_schedule': {
        'stage_1': {  # Epochs 0-5: Stabilize
            'epochs': 5,
            'weights': {
                'ce': 1.0,
                'align': 0.3,
                'bow': 0.1,
                'div': 0.2,
                'var': 0.05,
            }
        },
        'stage_2': {  # Epochs 5-10: Increase diversity
            'epochs': 5,
            'weights': {
                'ce': 1.0,
                'align': 0.5,
                'bow': 0.2,
                'div': 0.5,
                'var': 0.1,
            }
        },
        'stage_3': {  # Epochs 10-15: Push harder if needed
            'epochs': 5,
            'weights': {
                'ce': 1.0,
                'align': 0.7,
                'bow': 0.3,
                'div': 0.8,
                'var': 0.15,
            }
        },
        'stage_4': {  # Epochs 15+: Maintain or adapt
            'epochs': -1,  # Rest of training
            'weights': {
                'ce': 1.0,
                'align': 0.5,
                'bow': 0.2,
                'div': 0.3,
                'var': 0.1,
            }
        }
    },
    
    # Loss-specific settings
    'label_smoothing': 0.05,     # Standard smoothing
    'bow_vocab_size': 2000,      # Standard vocab
    'contrastive_tau': 0.07,     # Standard temperature
    
    # ============= BALANCED Generation Settings =============
    'generation': {
        'eval': {      
            'max_length': 16,
            'min_length': 4,
            'num_beams': 3,              # Use beam search for stability
            'temperature': 1.0,          # Standard temperature
            'repetition_penalty': 1.3,   # Moderate penalty
            'no_repeat_ngram_size': 3,   # Standard n-gram blocking
            'length_penalty': 1.0,
            'early_stopping': True,
        },
        'eval_sampling': {  # Alternative with sampling
            'max_length': 16,
            'min_length': 4,
            'do_sample': True,
            'temperature': 0.9,          # Slightly warm
            'top_p': 0.9,               # Nucleus sampling
            'top_k': 50,                # Limit vocabulary
            'repetition_penalty': 1.5,
            'no_repeat_ngram_size': 2,
        }
    },
    
    # ============= Data Configuration =============
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'num_workers': 0,
    
    # Moderate augmentation
    'augmentation': {
        'enabled': True,
        'noise_std': 0.01,           # Light noise
        'scale_range': (0.95, 1.05), # Light scaling
        'time_shift': 2,              # Light shifting
    },
    
    # ============= Monitoring and Logging =============
    'experiment_name': 'EEG-Chinese-BalancedRecovery',
    'log_interval': 20,
    'eval_interval': 1,
    'save_interval': 5,
    
    # Metrics to track
    'monitor_metrics': {
        'primary': 'val_bleu_4',           # Focus on quality
        'secondary': 'val_diversity_score', # But track diversity
        'early_stopping': 'val_bleu_4',
        'mode': 'max',
    },
    
    # ============= Advanced Settings =============
    'seed': 42,
    'deterministic': True,
    'mixed_precision': False,
    
    # Balanced thresholds
    'min_diversity_score': 0.2,  # Realistic minimum
    'max_diversity_score': 0.8,  # Realistic maximum
    'max_gradient_norm': 1.0,
    
    # ============= GRADUAL RECOVERY FLAGS =============
    'recovery_mode': True,        # Enables special recovery logic
    'noise_injection': {
        'enabled': True,
        'schedule': 'cosine',     # Smoother than linear
        'initial_std': 0.05,      # Much lighter noise
        'final_std': 0.0,         # No noise after recovery
        'decay_epochs': 10,
    },
    'teacher_forcing': {
        'enabled': True,
        'initial_ratio': 1.0,     # Full teacher forcing
        'decay_rate': 0.95,       # Gradual decay
        'min_ratio': 0.7,         # Never go too low
    },
    'gradient_accumulation_warmup': {
        'enabled': True,
        'initial_steps': 16,      # More accumulation initially
        'target_steps': 8,        # Normal accumulation
        'warmup_epochs': 5,
    },
}


def get_current_stage_weights(epoch):
    """
    Get loss weights for current training stage.
    
    Args:
        epoch: Current epoch number
        
    Returns:
        Dictionary of loss weights
    """
    if not CONFIG.get('staged_loss_schedule'):
        return CONFIG['loss_weights']
    
    cumulative_epochs = 0
    for stage_name, stage_config in CONFIG['staged_loss_schedule'].items():
        stage_epochs = stage_config['epochs']
        if stage_epochs == -1 or epoch < cumulative_epochs + stage_epochs:
            print(f"Using {stage_name} weights at epoch {epoch}")
            return stage_config['weights']
        cumulative_epochs += stage_epochs
    
    # Default to last stage
    return list(CONFIG['staged_loss_schedule'].values())[-1]['weights']


def get_optimizer_groups(model):
    """
    Create parameter groups with balanced learning rates.
    """
    groups = {
        'brain_encoder': [],
        'projection': [],
        'bart_decoder': [],
        'loss_heads': [],
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
            groups['projection'].append(param)
    
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


def get_noise_std_for_epoch(epoch):
    """
    Calculate noise standard deviation using cosine schedule.
    """
    if not CONFIG.get('noise_injection', {}).get('enabled', False):
        return 0.0
    
    noise_config = CONFIG['noise_injection']
    initial = noise_config['initial_std']
    final = noise_config['final_std']
    decay_epochs = noise_config['decay_epochs']
    
    if epoch >= decay_epochs:
        return final
    
    if noise_config['schedule'] == 'cosine':
        # Cosine annealing
        import math
        alpha = 0.5 * (1 + math.cos(math.pi * epoch / decay_epochs))
        return final + (initial - final) * alpha
    else:
        # Linear decay
        alpha = epoch / decay_epochs
        return initial * (1 - alpha) + final * alpha


def should_trigger_adaptation(metrics):
    """
    Balanced adaptation trigger.
    """
    diversity = metrics.get('diversity_score', 0.5)
    bleu = metrics.get('bleu_4', 0.0)
    
    # If completely collapsed
    if diversity < 0.1:
        return 'moderate_increase_diversity'
    
    # If generating gibberish (low BLEU, high diversity)
    if diversity > 0.7 and bleu < 1.0:
        return 'reduce_diversity_focus_quality'
    
    # Standard triggers
    if diversity < CONFIG['min_diversity_score']:
        return 'increase_diversity'
    elif diversity > CONFIG['max_diversity_score']:
        return 'decrease_diversity'
    
    return None


def update_loss_weights(current_weights, action, rate=None):
    """
    Balanced weight updates - avoid extremes.
    """
    rate = rate or CONFIG['adaptation_rate']
    weights = current_weights.copy()
    
    if action == 'moderate_increase_diversity':
        # Moderate increases to avoid gibberish
        weights['div'] = min(1.0, weights['div'] + 0.1)
        weights['align'] = min(1.0, weights['align'] + 0.1)
        weights['bow'] = min(0.5, weights['bow'] + 0.05)
        
    elif action == 'reduce_diversity_focus_quality':
        # Reduce diversity, increase CE for quality
        weights['div'] = max(0.1, weights['div'] * 0.7)
        weights['var'] = max(0.05, weights['var'] * 0.7)
        weights['ce'] = min(1.5, weights['ce'] * 1.1)
        
    elif action == 'increase_diversity':
        # Standard increase
        weights['div'] = min(1.0, weights['div'] * (1 + rate))
        weights['var'] = min(0.3, weights['var'] * (1 + rate))
        weights['align'] = min(1.0, weights['align'] * (1 + rate * 0.5))
        
    elif action == 'decrease_diversity':
        # Standard decrease
        weights['div'] = max(0.1, weights['div'] * (1 - rate))
        weights['var'] = max(0.05, weights['var'] * (1 - rate))
    
    return weights


def validate_config():
    """Validate configuration consistency."""
    assert CONFIG['train_split'] + CONFIG['val_split'] + CONFIG['test_split'] == 1.0
    assert CONFIG['batch_size'] > 0
    assert CONFIG['accumulation_steps'] > 0
    assert all(w >= 0 for w in CONFIG['loss_weights'].values())
    assert CONFIG['min_diversity_score'] < CONFIG['max_diversity_score']
    
    if CONFIG.get('recovery_mode', False):
        print("="*60)
        print("BALANCED RECOVERY MODE ACTIVATED")
        print("="*60)
        print("Gradual recovery strategy:")
        print("  - Stage 1 (0-5 epochs): Stabilize with light weights")
        print("  - Stage 2 (5-10 epochs): Gradually increase diversity")
        print("  - Stage 3 (10-15 epochs): Push harder if needed")
        print("  - Stage 4 (15+ epochs): Maintain balance")
        print(f"  - Noise injection: {CONFIG['noise_injection']['initial_std']} → 0")
        print(f"  - Teacher forcing: Enabled with gradual decay")
        print("="*60)
    
    return True


# Validate on import
if __name__ != "__main__":
    validate_config()