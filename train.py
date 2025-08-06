import os
import time
import torch
import wandb
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW
from dataset import EEGDataset
from evaluation import ChineseEvaluator
from models import EEGDecodingModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    'data_dir': 'data/eeg_data/',
    'montage_file': 'data/montage.csv',
    'pretrained_model': 'fnlp/bart-base-chinese',
    'hidden_dim': 768, 
    'n_timepoints': 1651,
    'max_length': 16,
    
    'disable_cross_region_attn': False, 
    'uniform_region_weight': False,    
    'cnn_only': False,                  
    'disable_cross_modal': False,     
    
    'epochs': 100,
    'batch_size': 4,          
    'accumulation_steps': 8, 
    'patience': 20,        
    'grad_clip_norm': 5.0,  
    'use_amp': False,       
    
    'brain_encoder_lr': 1e-4,   
    'bart_decoder_lr': 1e-5,  
    'bart_encoder_lr': 5e-6,     
    'projection_lr': 5e-4,       
    'warmup_steps': 1000,  
    'weight_decay': 0.01,
    'label_smoothing': 0.05,    
    
    'num_beams': 4,
    'max_gen_length': 20,
    'no_repeat_ngram_size': 2,
    'length_penalty': 1.0,
    'min_length': 3,
    'do_sample': True,
    'temperature': 0.8,
    'top_k': 50,
    
    'train_split': 0.8,
    'val_split': 0.1,
    'val_max_samples': 2000,     
    
    'experiment_name': 'EEG-Chinese',
    'save_dir': './checkpoints/',
    'log_interval': 50,        
    'eval_interval': 2,      
    'save_every_n_epochs': 5,
}

def setup_model_and_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['pretrained_model'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    dataset = EEGDataset(
        CONFIG['data_dir'], 
        CONFIG['montage_file'], 
        tokenizer, 
        max_length=CONFIG['max_length'],
        data_augmentation=True,  
    )
    
    stats = dataset.get_sample_stats()
    region_channel_counts = stats['region_channel_counts']
    
    model = EEGDecodingModel(
        n_timepoints=CONFIG['n_timepoints'],
        region_channel_counts=region_channel_counts,
        hidden_dim=CONFIG['hidden_dim'],
        disable_cross_region_attn=CONFIG['disable_cross_region_attn'],
        uniform_region_weight=CONFIG['uniform_region_weight'],
        cnn_only=CONFIG['cnn_only'],
        disable_cross_modal=CONFIG['disable_cross_modal'],
    ).to(device)
    
    initialize_weights(model)
    
    n = len(dataset)
    train_n = int(CONFIG['train_split'] * n)
    val_n = int(CONFIG['val_split'] * n)
    test_n = n - (train_n + val_n)
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    logger.info(f"Data splits - Train: {train_n}, Val: {val_n}, Test: {test_n}")
    
    return model, tokenizer, train_loader, val_loader, test_loader, device

def initialize_weights(model):
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

def setup_optimizer_and_scheduler(model, train_loader, config):
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
        param_groups.append({'params': brain_encoder_params, 'lr': config['brain_encoder_lr'], 'weight_decay': config['weight_decay']})
    if projection_params:
        param_groups.append({'params': projection_params, 'lr': config['projection_lr'], 'weight_decay': config['weight_decay']})
    if bart_encoder_params:
        param_groups.append({'params': bart_encoder_params, 'lr': config['bart_encoder_lr'], 'weight_decay': 0.0})
    if bart_decoder_params:
        param_groups.append({'params': bart_decoder_params, 'lr': config['bart_decoder_lr'], 'weight_decay': 0.0})
    
    optimizer = AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))
    
    total_steps = len(train_loader) * config['epochs'] // config['accumulation_steps']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=total_steps)
    
    return optimizer, scheduler

def safe_forward_pass(model, eeg, decoder_input_ids, labels, tokenizer, config):
    try:
        vocab_size = len(tokenizer.get_vocab())
        
        if decoder_input_ids.max() >= vocab_size or (labels != -100).any() and labels[labels != -100].max() >= vocab_size:
            return None
        
        outputs = model(
            eeg_data=eeg,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            label_smoothing_factor=config['label_smoothing'],
            use_cache=False
        )
        
        return outputs
        
    except RuntimeError as e:
        if "CUDA error" in str(e) or "out of memory" in str(e):
            torch.cuda.empty_cache()
            return None
        else:
            raise e

def train_epoch(model, train_loader, optimizer, scheduler, tokenizer, device, epoch, config):
    model.train()
    total_loss = 0.0
    total_samples = 0
    accumulation_step = 0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    
    for step, batch in enumerate(pbar):
        try:
            eeg = [region.to(device) for region in batch['eeg']]
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = safe_forward_pass(model, eeg, decoder_input_ids, labels, tokenizer, config)
            
            if outputs is None or outputs.loss is None:
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            
            loss = outputs.loss / config['accumulation_steps']
            
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
            
            loss.backward()
            accumulation_step += 1
            
            if accumulation_step >= config['accumulation_steps']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                accumulation_step = 0
            
            total_loss += outputs.loss.item() * len(labels)
            total_samples += len(labels)
            
            if step % config['log_interval'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f"{outputs.loss.item():.4f}", 'lr': f"{current_lr:.1e}"})
                
                wandb.log({
                    "train/step_loss": outputs.loss.item(),
                    "train/lr": current_lr,
                    "step": epoch * len(train_loader) + step
                })
                
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA error" in str(e):
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            else:
                raise e
    
    if accumulation_step > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    return avg_loss

def evaluate_model(model, val_loader, tokenizer, device, config):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            try:
                eeg = [region.to(device) for region in batch['eeg']]
                labels = batch['labels'].to(device)
                
                gen_config = {
                    'max_length': config['max_gen_length'],
                    'num_beams': config['num_beams'],
                    'length_penalty': config['length_penalty'],
                    'no_repeat_ngram_size': config['no_repeat_ngram_size'],
                    'min_length': config['min_length'],
                    'do_sample': False,
                    'pad_token_id': tokenizer.pad_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'use_cache': False
                }
                
                if config['num_beams'] > 1:
                    gen_config['early_stopping'] = True
                
                generated_ids = model.generate(eeg_data=eeg, **gen_config)
                
                for i in range(len(generated_ids)):
                    pred_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                    all_predictions.append(pred_text.strip())
                    
                    target_ids = labels[i][labels[i] != -100]
                    target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
                    all_targets.append(target_text.strip())
                        
            except Exception as e:
                torch.cuda.empty_cache()
                continue
    
    if not all_predictions:
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 'rouge_l_f': 0.0}
    
    try:
        evaluator = ChineseEvaluator()
        metrics = evaluator.compute_all_metrics(all_predictions, all_targets)
    except:
        metrics = {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 'rouge_l_f': 0.0}
    
    return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, save_path)

def main():
    wandb.init(project='EEG-Chinese', name=CONFIG['experiment_name'], config=CONFIG)
    
    model, tokenizer, train_loader, val_loader, test_loader, device = setup_model_and_data()
    optimizer, scheduler = setup_optimizer_and_scheduler(model, train_loader, CONFIG)
    
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    best_bleu4 = 0.0
    patience_counter = 0
    
    logger.info(f"Starting training for {CONFIG['epochs']} epochs...")
    
    for epoch in range(CONFIG['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, tokenizer, device, epoch, CONFIG)
        
        wandb.log({"train/epoch_loss": train_loss, "epoch": epoch})
        
        if (epoch + 1) % CONFIG['eval_interval'] == 0:
            val_metrics = evaluate_model(model, val_loader, tokenizer, device, CONFIG)
            
            wandb.log({
                "val/bleu_1": val_metrics['bleu_1'],
                "val/bleu_2": val_metrics['bleu_2'],
                "val/bleu_3": val_metrics['bleu_3'],
                "val/bleu_4": val_metrics['bleu_4'],
                "val/rouge_l": val_metrics['rouge_l_f'],
                "epoch": epoch
            })
            
            if val_metrics['bleu_4'] > best_bleu4:
                best_bleu4 = val_metrics['bleu_4']
                save_path = os.path.join(CONFIG['save_dir'], f"best_model.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, save_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= CONFIG['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % CONFIG['save_every_n_epochs'] == 0:
            save_path = os.path.join(CONFIG['save_dir'], f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, {}, save_path)
    
    logger.info(f"Training completed. Best BLEU-4: {best_bleu4:.3f}")
    wandb.finish()

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()