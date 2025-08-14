"""
Simplified training configuration.
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
    
    # ============= Training Parameters =============
    'epochs': 100,
    'batch_size': 4,
    'accumulation_steps': 8,
    'patience': 20,
    'grad_clip_norm': 1.0,
    
    # ============= Learning Rates =============
    'brain_encoder_lr': 3e-4,
    'bart_decoder_lr': 3e-5,
    'projection_lr': 1e-4,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    
    # ============= Generation Settings =============
    'generation': {
        'eval': {      
            'max_length': 16,
            'min_length': 4,
            'num_beams': 3,
            'early_stopping': True
        }
    },
    
    # ============= Data Configuration =============
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'num_workers': 0,
    
    # ============= Monitoring =============
    'log_interval': 20,
    'eval_interval': 1,
    'save_interval': 5,
    'seed': 42,
}


def get_optimizer_groups(model):
    groups = {
        'brain_encoder': [],
        'projection': [],
        'bart_decoder': [],
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'brain_encoder' in name:
            groups['brain_encoder'].append(param)
        elif 'eeg_to_bart' in name:
            groups['projection'].append(param)
        elif 'bart' in name:
            groups['bart_decoder'].append(param)
    
    return [
        {'params': groups['brain_encoder'], 'lr': CONFIG['brain_encoder_lr']},
        {'params': groups['projection'], 'lr': CONFIG['projection_lr']},
        {'params': groups['bart_decoder'], 'lr': CONFIG['bart_decoder_lr']},
    ]


def validate_config():
    assert CONFIG['train_split'] + CONFIG['val_split'] + CONFIG['test_split'] == 1.0
    return True