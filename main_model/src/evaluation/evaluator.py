"""
Evaluation metrics for Chinese text generation.
"""

import numpy as np
from collections import Counter
import jieba
import logging

logger = logging.getLogger(__name__)

try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Some evaluation metrics not available. Install nltk and rouge-score.")
    METRICS_AVAILABLE = False


class ChineseEvaluator:
    """
    Evaluator for Chinese text generation tasks.
    """
    
    def __init__(self):
        self.smoothing_function = SmoothingFunction().method1 if METRICS_AVAILABLE else None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False) if METRICS_AVAILABLE else None

    def tokenize_chinese(self, text):
        """Tokenize Chinese text using jieba."""
        if not text:
            return []
        return list(jieba.cut(text.strip()))

    def compute_bleu(self, predictions, references, n_gram=4):
        """Compute BLEU scores."""
        if not METRICS_AVAILABLE:
            return 0.0
            
        scores = []
        weights_dict = {
            1: (1.0, 0, 0, 0),
            2: (0.5, 0.5, 0, 0),
            3: (1/3, 1/3, 1/3, 0),
            4: (0.25, 0.25, 0.25, 0.25)
        }
        
        weights = weights_dict.get(n_gram, weights_dict[4])
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self.tokenize_chinese(pred)
            ref_tokens = [self.tokenize_chinese(ref)]
            
            if len(pred_tokens) == 0:
                scores.append(0.0)
                continue
                
            try:
                score = sentence_bleu(
                    ref_tokens, 
                    pred_tokens, 
                    weights=weights,
                    smoothing_function=self.smoothing_function
                )
                scores.append(score)
            except:
                scores.append(0.0)
        
        return np.mean(scores) * 100 if scores else 0.0

    def compute_rouge(self, predictions, references):
        """Compute ROUGE scores."""
        if not METRICS_AVAILABLE:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            # Join tokens with spaces for ROUGE computation
            pred_text = ' '.join(self.tokenize_chinese(pred))
            ref_text = ' '.join(self.tokenize_chinese(ref))
            
            try:
                scores = self.rouge_scorer.score(ref_text, pred_text)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            except:
                rouge_scores['rouge1'].append(0.0)
                rouge_scores['rouge2'].append(0.0)
                rouge_scores['rougeL'].append(0.0)
        
        return {
            'rouge1': np.mean(rouge_scores['rouge1']) * 100,
            'rouge2': np.mean(rouge_scores['rouge2']) * 100,
            'rougeL': np.mean(rouge_scores['rougeL']) * 100
        }

    def compute_exact_match(self, predictions, references):
        """Compute exact match accuracy."""
        if not predictions or not references:
            return 0.0
        
        matches = sum(1 for pred, ref in zip(predictions, references) 
                     if pred.strip() == ref.strip())
        return (matches / len(predictions)) * 100

    def compute_token_overlap(self, predictions, references):
        """Compute token-level overlap metrics."""
        if not predictions or not references:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        precisions = []
        recalls = []
        f1s = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self.tokenize_chinese(pred))
            ref_tokens = set(self.tokenize_chinese(ref))
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                precisions.append(1.0)
                recalls.append(1.0)
                f1s.append(1.0)
            elif len(pred_tokens) == 0:
                precisions.append(0.0)
                recalls.append(0.0)
                f1s.append(0.0)
            else:
                overlap = len(pred_tokens & ref_tokens)
                precision = overlap / len(pred_tokens)
                recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
        
        return {
            'precision': np.mean(precisions) * 100,
            'recall': np.mean(recalls) * 100,
            'f1': np.mean(f1s) * 100
        }

    def compute_all_metrics(self, predictions, references):
        """Compute all available metrics."""
        if not predictions or not references or len(predictions) != len(references):
            logger.warning(f"Invalid input: {len(predictions)} predictions, {len(references)} references")
            return self._empty_metrics()
        
        # Filter out empty predictions/references
        filtered_pairs = [(p, r) for p, r in zip(predictions, references) 
                         if p.strip() and r.strip()]
        
        if not filtered_pairs:
            logger.warning("No valid prediction-reference pairs found")
            return self._empty_metrics()
        
        filtered_predictions, filtered_references = zip(*filtered_pairs)
        
        metrics = {}
        
        # BLEU scores
        for n in [1, 2, 3, 4]:
            metrics[f'bleu_{n}'] = self.compute_bleu(
                filtered_predictions, filtered_references, n_gram=n
            )
        
        # ROUGE scores
        rouge_scores = self.compute_rouge(filtered_predictions, filtered_references)
        metrics.update({
            'rouge_1_f': rouge_scores['rouge1'],
            'rouge_2_f': rouge_scores['rouge2'],
            'rouge_l_f': rouge_scores['rougeL']
        })
        
        # Token overlap
        overlap_scores = self.compute_token_overlap(filtered_predictions, filtered_references)
        metrics.update({
            'token_precision': overlap_scores['precision'],
            'token_recall': overlap_scores['recall'],
            'token_f1': overlap_scores['f1']
        })
        
        # Exact match
        metrics['exact_match'] = self.compute_exact_match(filtered_predictions, filtered_references)
        
        # Length statistics
        pred_lengths = [len(self.tokenize_chinese(p)) for p in filtered_predictions]
        ref_lengths = [len(self.tokenize_chinese(r)) for r in filtered_references]
        
        metrics.update({
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0.0,
            'valid_pairs': len(filtered_pairs),
            'total_pairs': len(predictions)
        })
        
        return metrics

    def _empty_metrics(self):
        """Return empty metrics dict."""
        return {
            'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
            'rouge_1_f': 0.0, 'rouge_2_f': 0.0, 'rouge_l_f': 0.0,
            'token_precision': 0.0, 'token_recall': 0.0, 'token_f1': 0.0,
            'exact_match': 0.0, 'avg_pred_length': 0.0, 'avg_ref_length': 0.0,
            'length_ratio': 0.0, 'valid_pairs': 0, 'total_pairs': 0
        }