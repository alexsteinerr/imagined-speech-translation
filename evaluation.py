import os
import time
import torch
import wandb
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import jieba
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu
from transformers import BartConfig
from models import EEGDecodingModel

# Try to import rouge-chinese, fallback to rouge-score if not available
try:
    from rouge_chinese import Rouge
    ROUGE_CHINESE_AVAILABLE = True
except ImportError:
    try:
        from rouge_score import rouge_scorer
        ROUGE_CHINESE_AVAILABLE = False
        print("Warning: rouge-chinese not available, using rouge-score as fallback")
    except ImportError:
        print("Warning: No ROUGE library available")
        ROUGE_CHINESE_AVAILABLE = False


class ChineseEvaluator:
    """Specialized evaluator for Chinese text generation with enhanced compatibility"""
    
    def __init__(self):
        # Initialize Chinese-specific ROUGE scorer
        if ROUGE_CHINESE_AVAILABLE:
            self.rouge = Rouge()
        else:
            try:
                self.rouge = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], 
                    use_stemmer=False
                )
            except:
                self.rouge = None
                print("Warning: ROUGE scorer initialization failed")
        
        # Initialize BLEU scorers with error handling
        try:
            self.char_bleu = sacrebleu.metrics.BLEU(tokenize='char')
        except:
            print("Warning: Character BLEU scorer not available")
            self.char_bleu = None
        
        try:
            self.word_bleu = sacrebleu.metrics.BLEU(tokenize='zh')
        except:
            print("Warning: Word BLEU scorer not available")
            self.word_bleu = None
        
        # Character F-score
        try:
            self.chrf = sacrebleu.metrics.CHRF()
        except:
            print("Warning: chrF scorer not available")
            self.chrf = None
    
    def tokenize_chinese(self, text):
        """Tokenize Chinese text using jieba with error handling"""
        try:
            return ' '.join(jieba.cut(text.strip()))
        except:
            # Fallback to character-level tokenization
            return ' '.join(list(text.strip()))
    
    def compute_bleu_scores_sacrebleu(self, predictions, references):
        """Compute BLEU scores using sacrebleu with compatibility fixes"""
        bleu_metrics = {
            'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 
            'bleu_3': 0.0, 'bleu_4': 0.0, 'char_bleu': 0.0, 'word_bleu': 0.0
        }
        
        if not predictions or not references:
            return bleu_metrics
        
        try:
            # Overall BLEU (BLEU-4 by default)
            bleu_score = sacrebleu.corpus_bleu(predictions, [references], tokenize='zh')
            bleu_metrics['bleu'] = bleu_score.score
            bleu_metrics['bleu_4'] = bleu_score.score
            
            # Individual BLEU scores with different weights
            # BLEU-1
            try:
                bleu_1 = sacrebleu.corpus_bleu(
                    predictions, [references], 
                    tokenize='zh',
                    force=True,
                    lowercase=False
                )
                # Extract n-gram precisions manually for BLEU-1
                # This is a workaround since max_ngram_order may not be available
                bleu_metrics['bleu_1'] = bleu_1.precisions[0] if len(bleu_1.precisions) > 0 else 0.0
            except:
                bleu_metrics['bleu_1'] = 0.0
            
            # BLEU-2
            try:
                bleu_2 = sacrebleu.corpus_bleu(
                    predictions, [references], 
                    tokenize='zh',
                    force=True,
                    lowercase=False
                )
                # Calculate BLEU-2 manually from precisions
                if len(bleu_2.precisions) >= 2:
                    p1, p2 = bleu_2.precisions[0], bleu_2.precisions[1]
                    if p1 > 0 and p2 > 0:
                        bleu_metrics['bleu_2'] = bleu_2.bp * ((p1 * p2) ** 0.5)
                    else:
                        bleu_metrics['bleu_2'] = 0.0
                else:
                    bleu_metrics['bleu_2'] = 0.0
            except:
                bleu_metrics['bleu_2'] = 0.0
            
            # BLEU-3
            try:
                bleu_3 = sacrebleu.corpus_bleu(
                    predictions, [references], 
                    tokenize='zh',
                    force=True,
                    lowercase=False
                )
                # Calculate BLEU-3 manually from precisions
                if len(bleu_3.precisions) >= 3:
                    p1, p2, p3 = bleu_3.precisions[0], bleu_3.precisions[1], bleu_3.precisions[2]
                    if p1 > 0 and p2 > 0 and p3 > 0:
                        bleu_metrics['bleu_3'] = bleu_3.bp * ((p1 * p2 * p3) ** (1/3))
                    else:
                        bleu_metrics['bleu_3'] = 0.0
                else:
                    bleu_metrics['bleu_3'] = 0.0
            except:
                bleu_metrics['bleu_3'] = 0.0
            
            # Character-level BLEU
            if self.char_bleu:
                try:
                    char_bleu = self.char_bleu.corpus_score(predictions, [references])
                    bleu_metrics['char_bleu'] = char_bleu.score
                except:
                    bleu_metrics['char_bleu'] = 0.0
            
            # Word-level BLEU (jieba tokenization)
            if self.word_bleu:
                try:
                    tokenized_preds = [self.tokenize_chinese(p) for p in predictions]
                    tokenized_refs = [self.tokenize_chinese(r) for r in references]
                    word_bleu = sacrebleu.corpus_bleu(tokenized_preds, [tokenized_refs])
                    bleu_metrics['word_bleu'] = word_bleu.score
                except:
                    bleu_metrics['word_bleu'] = 0.0
            
        except Exception as e:
            print(f"BLEU computation error: {e}")
            # Return zero scores if computation fails
            
        return bleu_metrics
    
    def compute_rouge_scores_safe(self, predictions, references):
        """Compute ROUGE scores with fallback options"""
        rouge_metrics = {
            'rouge_1_f': 0.0, 'rouge_1_p': 0.0, 'rouge_1_r': 0.0,
            'rouge_2_f': 0.0, 'rouge_2_p': 0.0, 'rouge_2_r': 0.0,
            'rouge_l_f': 0.0, 'rouge_l_p': 0.0, 'rouge_l_r': 0.0
        }
        
        if not predictions or not references or not self.rouge:
            return rouge_metrics
        
        try:
            if ROUGE_CHINESE_AVAILABLE:
                # Use rouge-chinese
                rouge_scores = self.rouge.get_scores(
                    [' '.join(list(p)) for p in predictions],  # Character-level
                    [' '.join(list(r)) for r in references],
                    avg=True
                )
                
                rouge_metrics['rouge_1_f'] = rouge_scores['rouge-1']['f'] * 100
                rouge_metrics['rouge_1_p'] = rouge_scores['rouge-1']['p'] * 100
                rouge_metrics['rouge_1_r'] = rouge_scores['rouge-1']['r'] * 100
                
                rouge_metrics['rouge_2_f'] = rouge_scores['rouge-2']['f'] * 100
                rouge_metrics['rouge_2_p'] = rouge_scores['rouge-2']['p'] * 100
                rouge_metrics['rouge_2_r'] = rouge_scores['rouge-2']['r'] * 100
                
                rouge_metrics['rouge_l_f'] = rouge_scores['rouge-l']['f'] * 100
                rouge_metrics['rouge_l_p'] = rouge_scores['rouge-l']['p'] * 100
                rouge_metrics['rouge_l_r'] = rouge_scores['rouge-l']['r'] * 100
                
            else:
                # Use rouge-score as fallback
                rouge_1_f, rouge_1_p, rouge_1_r = [], [], []
                rouge_2_f, rouge_2_p, rouge_2_r = [], [], []
                rouge_l_f, rouge_l_p, rouge_l_r = [], [], []
                
                for pred, ref in zip(predictions, references):
                    # Add spaces between Chinese characters for ROUGE
                    pred_spaced = ' '.join(list(pred.strip()))
                    ref_spaced = ' '.join(list(ref.strip()))
                    
                    if pred_spaced and ref_spaced:
                        scores = self.rouge.score(ref_spaced, pred_spaced)
                        
                        rouge_1_f.append(scores['rouge1'].fmeasure)
                        rouge_1_p.append(scores['rouge1'].precision)
                        rouge_1_r.append(scores['rouge1'].recall)
                        
                        rouge_2_f.append(scores['rouge2'].fmeasure)
                        rouge_2_p.append(scores['rouge2'].precision)
                        rouge_2_r.append(scores['rouge2'].recall)
                        
                        rouge_l_f.append(scores['rougeL'].fmeasure)
                        rouge_l_p.append(scores['rougeL'].precision)
                        rouge_l_r.append(scores['rougeL'].recall)
                
                rouge_metrics['rouge_1_f'] = np.mean(rouge_1_f) * 100 if rouge_1_f else 0.0
                rouge_metrics['rouge_1_p'] = np.mean(rouge_1_p) * 100 if rouge_1_p else 0.0
                rouge_metrics['rouge_1_r'] = np.mean(rouge_1_r) * 100 if rouge_1_r else 0.0
                rouge_metrics['rouge_2_f'] = np.mean(rouge_2_f) * 100 if rouge_2_f else 0.0
                rouge_metrics['rouge_2_p'] = np.mean(rouge_2_p) * 100 if rouge_2_p else 0.0
                rouge_metrics['rouge_2_r'] = np.mean(rouge_2_r) * 100 if rouge_2_r else 0.0
                rouge_metrics['rouge_l_f'] = np.mean(rouge_l_f) * 100 if rouge_l_f else 0.0
                rouge_metrics['rouge_l_p'] = np.mean(rouge_l_p) * 100 if rouge_l_p else 0.0
                rouge_metrics['rouge_l_r'] = np.mean(rouge_l_r) * 100 if rouge_l_r else 0.0
                
        except Exception as e:
            print(f"ROUGE computation error: {e}")
            
        return rouge_metrics
    
    def compute_all_metrics(self, predictions, references):
        """Compute all metrics appropriate for Chinese with enhanced error handling"""
        metrics = {}
        
        # Ensure lists
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
        
        # Filter empty predictions/references
        valid_pairs = [
            (str(p).strip(), str(r).strip()) 
            for p, r in zip(predictions, references) 
            if str(p).strip() and str(r).strip()
        ]
        
        if not valid_pairs:
            print("Warning: No valid prediction-reference pairs found")
            return {metric: 0.0 for metric in [
                'bleu', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
                'char_bleu', 'word_bleu', 'chrf',
                'rouge_1_f', 'rouge_2_f', 'rouge_l_f',
                'char_precision', 'char_recall', 'char_f1',
                'exact_match', 'length_ratio'
            ]}
        
        preds, refs = zip(*valid_pairs)
        print(f"Computing metrics for {len(preds)} valid pairs...")
        
        # 1. BLEU scores with enhanced error handling
        bleu_metrics = self.compute_bleu_scores_sacrebleu(preds, refs)
        metrics.update(bleu_metrics)
        
        # 2. ROUGE scores with fallback
        rouge_metrics = self.compute_rouge_scores_safe(preds, refs)
        metrics.update(rouge_metrics)
        
        # 3. Character-level metrics
        char_metrics = self.compute_character_metrics(preds, refs)
        metrics.update(char_metrics)
        
        # 4. chrF score with error handling
        if self.chrf:
            try:
                chrf_score = self.chrf.corpus_score(preds, [refs])
                metrics['chrf'] = chrf_score.score
            except Exception as e:
                print(f"chrF computation error: {e}")
                metrics['chrf'] = 0.0
        else:
            metrics['chrf'] = 0.0
        
        # 5. Additional metrics
        try:
            metrics['exact_match'] = sum(p == r for p, r in zip(preds, refs)) / len(preds) * 100
            
            # Length statistics
            pred_lengths = [len(p) for p in preds]
            ref_lengths = [len(r) for r in refs]
            avg_pred_len = np.mean(pred_lengths)
            avg_ref_len = np.mean(ref_lengths)
            
            metrics['length_ratio'] = avg_pred_len / avg_ref_len if avg_ref_len > 0 else 1.0
            metrics['avg_pred_length'] = avg_pred_len
            metrics['avg_ref_length'] = avg_ref_len
            
        except Exception as e:
            print(f"Additional metrics computation error: {e}")
            metrics['exact_match'] = 0.0
            metrics['length_ratio'] = 1.0
            metrics['avg_pred_length'] = 0.0
            metrics['avg_ref_length'] = 0.0
        
        print(f"✓ Successfully computed {len(metrics)} metrics")
        return metrics
    
    def compute_character_metrics(self, predictions, references):
        """Compute detailed character-level metrics with error handling"""
        metrics = {
            'char_accuracy': 0.0, 'char_precision': 0.0, 
            'char_recall': 0.0, 'char_f1': 0.0, 'position_accuracy': 0.0
        }
        
        if not predictions or not references:
            return metrics
        
        try:
            total_chars_correct = 0
            total_chars_pred = 0
            total_chars_ref = 0
            
            position_accuracies = []
            char_precisions = []
            char_recalls = []
            
            for pred, ref in zip(predictions, references):
                pred_str = str(pred).strip()
                ref_str = str(ref).strip()
                
                # Position-based accuracy
                min_len = min(len(pred_str), len(ref_str))
                correct_positions = sum(1 for i in range(min_len) if pred_str[i] == ref_str[i])
                total_chars_correct += correct_positions
                total_chars_pred += len(pred_str)
                total_chars_ref += len(ref_str)
                
                if len(ref_str) > 0:
                    position_accuracies.append(correct_positions / len(ref_str))
                
                # Character set metrics
                pred_chars = set(list(pred_str))
                ref_chars = set(list(ref_str))
                
                if pred_chars:
                    precision = len(pred_chars & ref_chars) / len(pred_chars)
                    char_precisions.append(precision)
                
                if ref_chars:
                    recall = len(pred_chars & ref_chars) / len(ref_chars)
                    char_recalls.append(recall)
            
            # Overall metrics
            metrics['char_accuracy'] = total_chars_correct / total_chars_ref * 100 if total_chars_ref > 0 else 0
            metrics['char_precision'] = np.mean(char_precisions) * 100 if char_precisions else 0
            metrics['char_recall'] = np.mean(char_recalls) * 100 if char_recalls else 0
            metrics['position_accuracy'] = np.mean(position_accuracies) * 100 if position_accuracies else 0
            
            # F1 score
            if metrics['char_precision'] + metrics['char_recall'] > 0:
                metrics['char_f1'] = 2 * metrics['char_precision'] * metrics['char_recall'] / (
                    metrics['char_precision'] + metrics['char_recall']
                )
            else:
                metrics['char_f1'] = 0
            
        except Exception as e:
            print(f"Character metrics computation error: {e}")
        
        return metrics


def create_evaluation_plots(results_df, metrics, save_path, experiment_name):
    """Create comprehensive evaluation plots"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Score distributions
    ax1 = plt.subplot(3, 3, 1)
    results_df['bleu'].hist(bins=30, ax=ax1, alpha=0.7, color='blue')
    ax1.axvline(metrics['bleu'], color='red', linestyle='--', 
                label=f'Mean: {metrics["bleu"]:.2f}')
    ax1.set_xlabel('BLEU Score')
    ax1.set_ylabel('Count')
    ax1.set_title('BLEU Score Distribution')
    ax1.legend()
    
    ax2 = plt.subplot(3, 3, 2)
    results_df['char_f1'].hist(bins=30, ax=ax2, alpha=0.7, color='green')
    ax2.axvline(metrics['char_f1'], color='red', linestyle='--',
                label=f'Mean: {metrics["char_f1"]:.2f}')
    ax2.set_xlabel('Character F1 Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Character F1 Distribution')
    ax2.legend()
    
    ax3 = plt.subplot(3, 3, 3)
    results_df['rouge_l_f'].hist(bins=30, ax=ax3, alpha=0.7, color='orange')
    ax3.axvline(metrics['rouge_l_f'], color='red', linestyle='--',
                label=f'Mean: {metrics["rouge_l_f"]:.2f}')
    ax3.set_xlabel('ROUGE-L F1 Score')
    ax3.set_ylabel('Count')
    ax3.set_title('ROUGE-L F1 Distribution')
    ax3.legend()
    
    # 2. Length analysis
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(results_df['ref_length'], results_df['pred_length'], 
               alpha=0.5, s=10)
    ax4.plot([0, 50], [0, 50], 'r--', label='y=x')
    ax4.set_xlabel('Reference Length')
    ax4.set_ylabel('Predicted Length')
    ax4.set_title('Length Comparison')
    ax4.legend()
    max_len = max(results_df['ref_length'].max(), results_df['pred_length'].max()) + 5
    ax4.set_xlim(0, max_len)
    ax4.set_ylim(0, max_len)
    
    # 3. Score correlations
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(results_df['bleu'], results_df['char_f1'], alpha=0.5, s=10)
    ax5.set_xlabel('BLEU Score')
    ax5.set_ylabel('Character F1')
    ax5.set_title('BLEU vs Character F1')
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(results_df['bleu'], results_df['rouge_l_f'], alpha=0.5, s=10)
    ax6.set_xlabel('BLEU Score')
    ax6.set_ylabel('ROUGE-L F1')
    ax6.set_title('BLEU vs ROUGE-L')
    
    # 4. Score by length
    ax7 = plt.subplot(3, 3, 7)
    length_bins = pd.cut(results_df['ref_length'], bins=5)
    length_scores = results_df.groupby(length_bins)['bleu'].mean()
    length_scores.plot(kind='bar', ax=ax7)
    ax7.set_xlabel('Reference Length Bins')
    ax7.set_ylabel('Average BLEU Score')
    ax7.set_title('BLEU Score by Reference Length')
    ax7.tick_params(axis='x', rotation=45)
    
    # 5. Worst and best examples
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    worst_idx = results_df['bleu'].idxmin()
    best_idx = results_df['bleu'].idxmax()
    
    text = f"Best Example (BLEU={results_df.loc[best_idx, 'bleu']:.2f}):\n"
    text += f"Pred: {results_df.loc[best_idx, 'prediction'][:50]}...\n"
    text += f"Ref:  {results_df.loc[best_idx, 'reference'][:50]}...\n\n"
    text += f"Worst Example (BLEU={results_df.loc[worst_idx, 'bleu']:.2f}):\n"
    text += f"Pred: {results_df.loc[worst_idx, 'prediction'][:50]}...\n"
    text += f"Ref:  {results_df.loc[worst_idx, 'reference'][:50]}..."
    
    ax8.text(0.1, 0.5, text, fontsize=8, verticalalignment='center',
             fontfamily='monospace', wrap=True)
    ax8.set_title('Best and Worst Examples')
    
    # 6. Metric summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    metric_names = ['BLEU', 'BLEU-1', 'BLEU-4', 'Char F1', 'ROUGE-L F1', 'chrF']
    metric_values = [
        metrics['bleu'], metrics['bleu_1'], metrics['bleu_4'],
        metrics['char_f1'], metrics['rouge_l_f'], metrics['chrf']
    ]
    
    y_pos = np.arange(len(metric_names))
    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
    
    for i, (name, value) in enumerate(zip(metric_names, metric_values)):
        ax9.text(0.1, 0.9 - i*0.15, f"{name}: {value:.2f}", 
                fontsize=14, fontweight='bold', color=colors[i])
    
    ax9.set_title(f'{experiment_name} - Metric Summary', fontsize=16)
    
    plt.suptitle(f'Evaluation Results - {experiment_name}', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'evaluation_plots.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_complete_test_set(model, test_loader, tokenizer, device, save_dir, 
                             experiment_name="baseline", chinese_eval=None, config=None):
    """Evaluate model on complete test set and save all results"""
    
    if chinese_eval is None:
        chinese_eval = ChineseEvaluator()
    
    print(f"\n=== Evaluating {experiment_name} on Complete Test Set ===")
    
    model.eval()
    all_predictions = []
    all_references = []
    all_eeg_features = []
    sample_metadata = []
    
    # Collect all predictions
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating predictions")):
            try:
                # Move data to device
                eeg = [region.to(device) for region in batch['eeg']]
                labels = batch['labels'].squeeze(1).to(device)
                
                # Get EEG features
                eeg_feats = model.brain_encoder(eeg)
                all_eeg_features.append(eeg_feats.cpu().numpy())
                
                # Generate predictions
                generated_ids = model.bart_decoder.generate_from_eeg(
                    eeg_feats,
                    max_length=config['max_length'] if config else 32,
                    num_beams=config['num_beams'] if config else 4,
                    do_sample=config['do_sample'] if config else True,
                    temperature=config['temperature'] if config else 0.8,
                    top_p=config['top_p'] if config else 0.9,
                    top_k=config['top_k'] if config else 50,
                    no_repeat_ngram_size=config['no_repeat_ngram_size'] if config else 3,
                    length_penalty=config['length_penalty'] if config else 1.0
                )
                
                # Decode texts
                for i in range(len(generated_ids)):
                    # Prediction
                    pred_text = tokenizer.decode(
                        generated_ids[i], 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=True
                    ).strip()
                    
                    # Reference
                    valid_labels = labels[i][labels[i] != -100]
                    ref_text = tokenizer.decode(
                        valid_labels, 
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    ).strip()
                    
                    all_predictions.append(pred_text)
                    all_references.append(ref_text)
                    
                    # Metadata
                    sample_metadata.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'global_idx': batch_idx * test_loader.batch_size + i
                    })
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Compute all metrics
    print("Computing comprehensive metrics...")
    metrics = chinese_eval.compute_all_metrics(all_predictions, all_references)
    
    # Compute per-sample metrics
    print("Computing per-sample scores...")
    per_sample_scores = []
    
    for pred, ref, meta in zip(all_predictions, all_references, sample_metadata):
        try:
            sample_metrics = chinese_eval.compute_all_metrics([pred], [ref])
            sample_metrics['prediction'] = pred
            sample_metrics['reference'] = ref
            sample_metrics['pred_length'] = len(pred)
            sample_metrics['ref_length'] = len(ref)
            sample_metrics.update(meta)
            per_sample_scores.append(sample_metrics)
        except Exception as e:
            print(f"Error computing sample metrics: {e}")
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(per_sample_scores)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{experiment_name}_results_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Save detailed results CSV
    results_df.to_csv(
        os.path.join(save_path, "detailed_results.csv"), 
        index=False, 
        encoding='utf-8-sig'
    )
    
    # 2. Save summary metrics
    summary = {
        'experiment': experiment_name,
        'timestamp': timestamp,
        'num_samples': len(all_predictions),
        'metrics': metrics,
        'config': config
    }
    
    with open(os.path.join(save_path, "summary_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 3. Save aggregated metrics CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df['experiment'] = experiment_name
    metrics_df.to_csv(
        os.path.join(save_path, "aggregated_metrics.csv"), 
        index=False
    )
    
    # 4. Save predictions and references for manual inspection
    inspection_df = pd.DataFrame({
        'prediction': all_predictions,
        'reference': all_references,
        'pred_length': [len(p) for p in all_predictions],
        'ref_length': [len(r) for r in all_references],
    })
    inspection_df.to_csv(
        os.path.join(save_path, "predictions_for_inspection.csv"),
        index=False,
        encoding='utf-8-sig'
    )
    
    # 5. Create visualizations
    try:
        create_evaluation_plots(results_df, metrics, save_path, experiment_name)
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    # Print summary
    print(f"\n=== {experiment_name} Results Summary ===")
    print(f"Total samples evaluated: {len(all_predictions)}")
    print(f"\nKey Metrics:")
    print(f"  BLEU: {metrics['bleu']:.2f}")
    print(f"  BLEU-1: {metrics['bleu_1']:.2f}")
    print(f"  BLEU-4: {metrics['bleu_4']:.2f}")
    print(f"  Character BLEU: {metrics['char_bleu']:.2f}")
    print(f"  Word BLEU: {metrics['word_bleu']:.2f}")
    print(f"  chrF: {metrics['chrf']:.2f}")
    print(f"  ROUGE-1 F1: {metrics['rouge_1_f']:.2f}")
    print(f"  ROUGE-2 F1: {metrics['rouge_2_f']:.2f}")
    print(f"  ROUGE-L F1: {metrics['rouge_l_f']:.2f}")
    print(f"  Character F1: {metrics['char_f1']:.2f}")
    print(f"  Character Accuracy: {metrics['char_accuracy']:.2f}%")
    print(f"  Exact Match: {metrics['exact_match']:.2f}%")
    print(f"  Length Ratio: {metrics['length_ratio']:.2f}")
    
    print(f"\nResults saved to: {save_path}")
    
    return metrics, results_df, save_path


def run_ablation_study(base_model_path, test_loader, tokenizer, device, save_dir, config):
    """Run comprehensive ablation study with enhanced error handling"""
    
    print("\n=== Running Ablation Study ===")
    
    # Define ablation configurations
    ablation_configs = [
        {
            'name': 'baseline',
            'config': {
                'disable_cross_region_attn': False,
                'uniform_region_weight': False,
                'disable_positional_encoding': False,
                'cnn_only': False,
                'disable_cross_modal': False,
            }
        },
        {
            'name': 'no_cross_region_attn',
            'config': {
                'disable_cross_region_attn': True,
                'uniform_region_weight': False,
                'disable_positional_encoding': False,
                'cnn_only': False,
                'disable_cross_modal': False,
            }
        },
        {
            'name': 'uniform_region_weight',
            'config': {
                'disable_cross_region_attn': False,
                'uniform_region_weight': True,
                'disable_positional_encoding': False,
                'cnn_only': False,
                'disable_cross_modal': False,
            }
        },
        {
            'name': 'no_positional_encoding',
            'config': {
                'disable_cross_region_attn': False,
                'uniform_region_weight': False,
                'disable_positional_encoding': True,
                'cnn_only': False,
                'disable_cross_modal': False,
            }
        },
        {
            'name': 'cnn_only',
            'config': {
                'disable_cross_region_attn': False,
                'uniform_region_weight': False,
                'disable_positional_encoding': False,
                'cnn_only': True,
                'disable_cross_modal': False,
            }
        },
        {
            'name': 'no_cross_modal',
            'config': {
                'disable_cross_region_attn': False,
                'uniform_region_weight': False,
                'disable_positional_encoding': False,
                'cnn_only': False,
                'disable_cross_modal': True,
            }
        },
        {
            'name': 'minimal_model',
            'config': {
                'disable_cross_region_attn': True,
                'uniform_region_weight': True,
                'disable_positional_encoding': True,
                'cnn_only': True,
                'disable_cross_modal': True,
            }
        }
    ]
    
    # Initialize evaluator
    chinese_eval = ChineseEvaluator()
    
    # Results storage
    all_results = []
    ablation_save_dir = os.path.join(save_dir, 'ablation_study')
    os.makedirs(ablation_save_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {base_model_path}")
    try:
        checkpoint = torch.load(base_model_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return []
    
    # Run each ablation
    for ablation in ablation_configs:
        print(f"\n--- Evaluating: {ablation['name']} ---")
        
        try:
            # Create model with ablation config
            bart_config = BartConfig.from_pretrained(config['pretrained_model'])
            
            model = EEGDecodingModel(
                n_timepoints=config['n_timepoints'],
                hidden_dim=config['hidden_dim'],
                bart_config=bart_config,
                **ablation['config']
            ).to(device)
            
            # Load weights (will have mismatches for different architectures)
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"  ✓ Loaded weights for {ablation['name']}")
            except Exception as e:
                print(f"  Warning: Could not load all weights for {ablation['name']}: {e}")
            
            # Evaluate
            metrics, results_df, result_path = evaluate_complete_test_set(
                model, test_loader, tokenizer, device, 
                ablation_save_dir, ablation['name'], chinese_eval, config
            )
            
            # Store results
            result_summary = {
                'name': ablation['name'],
                'config': ablation['config'],
                'path': result_path,
                **metrics
            }
            all_results.append(result_summary)
            
            # Clean up model
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error evaluating {ablation['name']}: {e}")
            continue
    
    # Create comparison plots
    if all_results:
        try:
            create_ablation_comparison_plots(all_results, ablation_save_dir)
        except Exception as e:
            print(f"Error creating comparison plots: {e}")
        
        # Save comparison CSV
        comparison_df = pd.DataFrame(all_results)
        comparison_df.to_csv(
            os.path.join(ablation_save_dir, 'ablation_comparison.csv'),
            index=False
        )
        
        # Print comparison table
        print("\n=== Ablation Study Results ===")
        print(comparison_df[['name', 'bleu', 'bleu_4', 'char_f1', 'rouge_l_f', 'exact_match']].to_string())
    
    return all_results


def create_ablation_comparison_plots(results, save_dir):
    """Create comparison plots for ablation study"""
    
    # Prepare data
    df = pd.DataFrame(results)
    
    # Key metrics to compare
    metrics_to_plot = ['bleu', 'bleu_4', 'char_f1', 'rouge_l_f', 'char_accuracy', 'chrf']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Sort by metric value
        sorted_df = df.sort_values(metric, ascending=True)
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_df)))
        bars = ax.barh(sorted_df['name'], sorted_df[metric], color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}', ha='left', va='center')
        
        ax.set_xlabel(metric.upper().replace('_', ' '))
        ax.set_title(f'{metric.upper().replace("_", " ")} Comparison')
        ax.set_xlim(0, max(sorted_df[metric]) * 1.1)
    
    plt.suptitle('Ablation Study Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create heatmap of all metrics
    metrics_matrix = df.set_index('name')[metrics_to_plot].T
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Score'})
    plt.title('Ablation Study - All Metrics Heatmap')
    plt.xlabel('Model Configuration')
    plt.ylabel('Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_heatmap.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def log_metrics_to_wandb(metrics, prefix="val", epoch=None, additional_metrics=None):
    """Log metrics to WandB with organized structure"""
    
    wandb_metrics = {}
    
    # Add epoch if provided
    if epoch is not None:
        wandb_metrics["epoch"] = epoch
    
    # Add additional metrics if provided
    if additional_metrics:
        wandb_metrics.update(additional_metrics)
    
    # Add all metrics with proper prefixes
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            if metric_name in ['bleu', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'char_bleu', 'word_bleu']:
                wandb_metrics[f"{prefix}/bleu/{metric_name}"] = value
            elif metric_name in ['rouge_1_f', 'rouge_2_f', 'rouge_l_f', 'rouge_1_p', 'rouge_2_p', 'rouge_l_p', 'rouge_1_r', 'rouge_2_r', 'rouge_l_r']:
                wandb_metrics[f"{prefix}/rouge/{metric_name}"] = value
            elif metric_name in ['char_accuracy', 'char_precision', 'char_recall', 'char_f1', 'position_accuracy']:
                wandb_metrics[f"{prefix}/character/{metric_name}"] = value
            elif metric_name in ['chrf', 'exact_match', 'length_ratio', 'avg_pred_length', 'avg_ref_length']:
                wandb_metrics[f"{prefix}/other/{metric_name}"] = value
            else:
                wandb_metrics[f"{prefix}/{metric_name}"] = value
    
    try:
        wandb.log(wandb_metrics)
    except Exception as e:
        print(f"Error logging to wandb: {e}")