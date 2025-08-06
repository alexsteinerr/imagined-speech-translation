import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import sacrebleu
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoTokenizer
import jieba
import re

class ChineseSemanticLoss(nn.Module):
    """Enhanced semantic loss specifically designed for Chinese text generation"""
    
    def __init__(self, hidden_dim: int = 768, device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        
        # Use a more powerful Chinese BERT model for better semantic understanding
        self.model_name = "hfl/chinese-roberta-wwm-ext-large"
        try:
            print(f"Loading {self.model_name} for Chinese semantic similarity...")
            self.encoder = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.embed_dim = 1024  # Large model has 1024 dimensions
            
            # Freeze BERT to save memory and computation
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        except Exception as e:
            print(f"Failed to load Chinese RoBERTa: {e}")
            print("Falling back to Chinese BERT...")
            try:
                self.model_name = "hfl/chinese-bert-wwm-ext"
                self.encoder = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.embed_dim = 768
            except Exception as e2:
                print(f"Failed to load Chinese BERT: {e2}")
                print("Using multilingual sentence transformer...")
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    device=str(self.device)
                )
                self.embed_dim = 384
                self.use_sentence_transformer = True
            else:
                self.use_sentence_transformer = False
        else:
            self.use_sentence_transformer = False
        
        # Enhanced projection with residual connection
        if self.hidden_dim != self.embed_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ).to(self.device)
        else:
            self.projection = nn.Identity()
        
        # Temperature parameter for similarity computation
        self.temperature = nn.Parameter(torch.ones([]) * 0.1)
        
        # Initialize jieba for Chinese word segmentation
        jieba.initialize()
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings with enhanced Chinese processing"""
        # Clean and normalize Chinese texts
        cleaned_texts = []
        for text in texts:
            text = text.strip()
            if not text:
                text = "空"
            # Remove excessive whitespace and normalize
            text = re.sub(r'\s+', '', text)  # Remove all whitespace
            cleaned_texts.append(text)
        
        if self.use_sentence_transformer:
            embeddings = self.encoder.encode(
                cleaned_texts, convert_to_tensor=True, 
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self.device
            )
            return embeddings.to(self.device)
        else:
            # Enhanced BERT encoding with better tokenization
            inputs = self.tokenizer(
                cleaned_texts, padding=True, truncation=True, 
                max_length=128, return_tensors='pt'  # Increased max length
            ).to(self.device)
            
            outputs = self.encoder(**inputs)
            # Use mean pooling instead of just CLS token for better representation
            attention_mask = inputs['attention_mask']
            embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            embeddings = embeddings / attention_mask.sum(dim=1, keepdim=True)
            return F.normalize(embeddings, p=2, dim=1)
    
    def forward(self, eeg_features: torch.Tensor, pred_texts: List[str], 
                target_texts: List[str]) -> Tuple[torch.Tensor, float]:
        """
        Compute enhanced semantic similarity loss for Chinese
        Returns: (loss, similarity_score)
        """
        # Handle empty batches
        if len(pred_texts) == 0 or len(target_texts) == 0:
            return torch.tensor(0.0, device=self.device), 0.0

        # EEG features should already be (B, hidden_dim) from the model
        if len(eeg_features.shape) > 2:
            eeg_features = eeg_features.mean(dim=1)
        
        # Project EEG features with enhanced projection
        eeg_proj = self.projection(eeg_features)
        eeg_proj = F.normalize(eeg_proj, p=2, dim=1)
        
        # Get text embeddings (no gradients)
        with torch.no_grad():
            pred_embeds = self.encode_texts(pred_texts)
            target_embeds = self.encode_texts(target_texts)
        
        # Ensure batch consistency
        min_batch = min(eeg_proj.size(0), pred_embeds.size(0), target_embeds.size(0))
        if min_batch == 0:
            return torch.tensor(0.0, device=self.device), 0.0
            
        eeg_proj = eeg_proj[:min_batch]
        pred_embeds = pred_embeds[:min_batch]
        target_embeds = target_embeds[:min_batch]
        
        # Compute similarities with temperature scaling
        pred_target_sim = F.cosine_similarity(pred_embeds, target_embeds, dim=1) / self.temperature
        eeg_target_sim = F.cosine_similarity(eeg_proj, target_embeds, dim=1) / self.temperature
        
        # Enhanced loss: maximize similarity between EEG and target
        # Add contrastive component to push EEG closer to target than to prediction
        loss = (1 - eeg_target_sim).mean() + 0.1 * F.relu(pred_target_sim - eeg_target_sim).mean()
        
        # Average similarity for monitoring
        avg_sim = (pred_target_sim.mean() + eeg_target_sim.mean()) / 2
        
        return loss, avg_sim.item()

class ChineseGrammaticalLoss(nn.Module):
    """Grammatical correctness loss for Chinese text"""
    
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Common Chinese grammatical patterns
        self.common_patterns = [
            r'的',  # Possessive particle
            r'了',  # Perfective aspect
            r'着',  # Progressive aspect
            r'过',  # Experiential aspect
            r'在',  # Location/Progressive
            r'是',  # Copula
            r'有',  # Have/Exist
            r'不',  # Negation
            r'没',  # Negation
            r'吗',  # Question particle
            r'呢',  # Question particle
            r'吧',  # Suggestion particle
        ]
        
        # Compile patterns
        self.patterns = [re.compile(pattern) for pattern in self.common_patterns]
    
    def forward(self, pred_texts: List[str], target_texts: List[str]) -> torch.Tensor:
        """
        Compute grammatical correctness loss
        """
        batch_size = len(pred_texts)
        if batch_size == 0:
            return torch.tensor(0.0, device=self.device)
        
        penalties = []
        
        for pred, target in zip(pred_texts, target_texts):
            penalty = 0.0
            
            # Skip empty texts
            if not pred.strip() or not target.strip():
                penalties.append(1.0)
                continue
            
            try:
                # 1. Check for basic Chinese character patterns
                pred_chars = set(pred)
                target_chars = set(target)
                
                # Penalty for missing important characters from target
                important_chars = target_chars - pred_chars
                if important_chars:
                    penalty += len(important_chars) / len(target_chars) * 0.3
                
                # 2. Check for grammatical particle usage
                pred_particles = sum(1 for pattern in self.patterns if pattern.search(pred))
                target_particles = sum(1 for pattern in self.patterns if pattern.search(target))
                
                if target_particles > 0:
                    particle_diff = abs(pred_particles - target_particles) / target_particles
                    penalty += particle_diff * 0.2
                
                # 3. Check for proper sentence ending
                if target and target[-1] in '。！？':
                    if not pred or pred[-1] not in '。！？':
                        penalty += 0.1
                
                # 4. Check for balanced parentheses and quotes
                if target.count('（') != target.count('）') or target.count('"') % 2 != 0:
                    if pred.count('（') == pred.count('）') and pred.count('"') % 2 == 0:
                        penalty += 0.1  # Reward for fixing target's imbalance
            except Exception as e:
                print(f"Grammatical loss error: {e}")
                penalty = 1.0
            
            penalties.append(min(penalty, 1.0))
        
        return torch.tensor(penalties, device=self.device).mean()

class ChineseStructuralLoss(nn.Module):
    """Enhanced structural loss for Chinese text quality"""
    
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_preferred_length = 3
        self.max_preferred_length = 50
        
        # Chinese-specific structural rules
        self.common_words = set([
            '的', '了', '在', '是', '有', '不', '没', '和', '与', '或', '但', '而', '因为', '所以',
            '如果', '虽然', '但是', '然后', '首先', '最后', '总之', '例如', '比如', '关于'
        ])
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, 
                pred_texts: List[str]) -> torch.Tensor:
        """
        Compute enhanced structural quality loss for Chinese
        """
        batch_size = len(pred_texts)
        penalties = []
        
        for i in range(batch_size):
            penalty = 0.0
            text = pred_texts[i].strip()
            
            if not text:
                penalties.append(1.0)
                continue
            
            try:
                # 1. Length penalty (more lenient for Chinese)
                text_len = len(text)
                if text_len < self.min_preferred_length:
                    penalty += (self.min_preferred_length - text_len) / self.min_preferred_length * 0.5
                elif text_len > self.max_preferred_length:
                    penalty += (text_len - self.max_preferred_length) / self.max_preferred_length * 0.3
                
                # 2. Character repetition check (more sophisticated)
                if len(text) > 2:
                    # Check for repeated characters (allow some repetition for emphasis)
                    repeated_chars = 0
                    for j in range(1, len(text)):
                        if text[j] == text[j-1]:
                            repeated_chars += 1
                            # Allow up to 2 consecutive repetitions
                            if j > 1 and text[j] == text[j-2]:
                                repeated_chars += 0.5
                    
                    repetition_ratio = repeated_chars / len(text)
                    if repetition_ratio > 0.1:  # More lenient threshold
                        penalty += repetition_ratio * 0.3
                
                # 3. Word-level diversity (using jieba segmentation)
                try:
                    words = list(jieba.cut(text))
                    if len(words) > 1:
                        unique_words = len(set(words))
                        diversity_ratio = unique_words / len(words)
                        if diversity_ratio < 0.7:  # Encourage word diversity
                            penalty += (0.7 - diversity_ratio) * 0.2
                except:
                    pass  # Skip if jieba fails
                
                # 4. Check for balanced structure
                # Count common structural words
                structural_words = sum(1 for word in self.common_words if word in text)
                if len(text) > 10 and structural_words == 0:
                    penalty += 0.1  # Encourage use of structural words for longer texts
            except Exception as e:
                print(f"Structure loss error: {e}")
                penalty = 1.0
            
            penalties.append(min(penalty, 1.0))
        
        return torch.tensor(penalties, device=self.device).mean()

class ChineseFluencyLoss(nn.Module):
    """Fluency loss to encourage natural Chinese text flow"""
    
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Common Chinese character combinations (bigrams)
        self.common_bigrams = set([
            '我们', '你们', '他们', '它们', '这个', '那个', '这些', '那些',
            '什么', '怎么', '为什么', '因为', '所以', '但是', '然后', '首先',
            '最后', '总之', '例如', '比如', '关于', '对于', '由于', '因此'
        ])
    
    def forward(self, pred_texts: List[str]) -> torch.Tensor:
        """
        Compute fluency loss based on character combinations
        """
        batch_size = len(pred_texts)
        if batch_size == 0:
            return torch.tensor(0.0, device=self.device)
        
        penalties = []
        
        for text in pred_texts:
            text = text.strip()
            if not text:
                penalties.append(1.0)
                continue
            
            penalty = 0.0
            
            try:
                # 1. Check for common bigram usage (reward natural combinations)
                bigrams = [text[i:i+2] for i in range(len(text)-1)]
                common_bigram_count = sum(1 for bg in bigrams if bg in self.common_bigrams)
                
                if len(bigrams) > 0:
                    common_ratio = common_bigram_count / len(bigrams)
                    # Small penalty if too few common bigrams (unnatural)
                    if common_ratio < 0.1 and len(text) > 4:
                        penalty += (0.1 - common_ratio) * 0.2
                
                # 2. Check for character spacing and flow
                # Penalize excessive punctuation
                punct_count = sum(1 for char in text if char in '，。！？；：')
                if len(text) > 0:
                    punct_ratio = punct_count / len(text)
                    if punct_ratio > 0.3:  # Too much punctuation
                        penalty += (punct_ratio - 0.3) * 0.3
            except Exception as e:
                print(f"Fluency loss error: {e}")
                penalty = 1.0
            
            penalties.append(min(penalty, 1.0))
        
        return torch.tensor(penalties, device=self.device).mean()

class ModelSpecificLoss(nn.Module):
    """Enhanced loss function designed specifically for Chinese brain-to-text generation"""
    
    def __init__(self, vocab_size: int, hidden_dim: int = 768, device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize enhanced loss components with device
        self.semantic_loss = ChineseSemanticLoss(hidden_dim, self.device)
        self.grammatical_loss = ChineseGrammaticalLoss(self.device)
        self.structure_loss = ChineseStructuralLoss(self.device)
        self.fluency_loss = ChineseFluencyLoss(self.device)
        
        # Metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
        
        print("Enhanced Chinese-specific loss functions initialized")
        print(f"  - Semantic loss with dimension: {hidden_dim}")
        print(f"  - Grammatical correctness loss")
        print(f"  - Structural quality loss")
        print(f"  - Fluency loss")
    
    def forward(self, 
                base_loss: torch.Tensor,
                eeg_features: torch.Tensor,
                logits: torch.Tensor,
                labels: torch.Tensor,
                pred_texts: List[str],
                target_texts: List[str],
                weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted combination of enhanced losses for Chinese text generation
        """
        loss_dict = {'base': base_loss.item()}
        
        # Start with weighted base loss
        total_loss = base_loss * weights['base']
        
        # Skip auxiliary losses if no valid predictions
        valid_preds = [p for p in pred_texts if p.strip()]
        if not valid_preds:
            return total_loss, loss_dict
        
        # 1. Enhanced semantic similarity loss
        try:
            semantic_loss, semantic_sim = self.semantic_loss(
                eeg_features, pred_texts, target_texts
            )
            total_loss = total_loss + semantic_loss * weights.get('semantic', 0.15)
            loss_dict['semantic'] = semantic_loss.item()
            loss_dict['semantic_sim'] = semantic_sim
        except Exception as e:
            print(f"Semantic loss error: {e}")
            loss_dict['semantic'] = 0.0
            loss_dict['semantic_sim'] = 0.0
        
        # 2. Grammatical correctness loss
        try:
            grammatical_loss = self.grammatical_loss(pred_texts, target_texts)
            total_loss = total_loss + grammatical_loss * weights.get('grammatical', 0.10)
            loss_dict['grammatical'] = grammatical_loss.item()
        except Exception as e:
            print(f"Grammatical loss error: {e}")
            loss_dict['grammatical'] = 0.0
        
        # 3. Enhanced structural loss
        try:
            structure_loss = self.structure_loss(logits, labels, pred_texts)
            total_loss = total_loss + structure_loss * weights.get('structure', 0.08)
            loss_dict['structure'] = structure_loss.item()
        except Exception as e:
            print(f"Structure loss error: {e}")
            loss_dict['structure'] = 0.0
        
        # 4. Fluency loss
        try:
            fluency_loss = self.fluency_loss(pred_texts)
            total_loss = total_loss + fluency_loss * weights.get('fluency', 0.05)
            loss_dict['fluency'] = fluency_loss.item()
        except Exception as e:
            print(f"Fluency loss error: {e}")
            loss_dict['fluency'] = 0.0
        
        return total_loss, loss_dict
    
    def compute_metrics(self, pred_texts: List[str], target_texts: List[str]) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics for Chinese text"""
        # Filter valid pairs
        valid_pairs = [
            (p.strip(), t.strip()) 
            for p, t in zip(pred_texts, target_texts) 
            if p.strip() and t.strip()
        ]
        
        if not valid_pairs:
            return {
                'bleu': 0.0, 
                'rouge1': 0.0, 
                'rougeL': 0.0, 
                'semantic_sim': 0.0, 
                'char_accuracy': 0.0,
                'grammatical_score': 0.0,
                'fluency_score': 0.0,
                'avg_pred_len': 0.0,
                'avg_target_len': 0.0
            }
        
        preds, targets = zip(*valid_pairs)
        
        # BLEU score
        try:
            bleu = sacrebleu.corpus_bleu(
                list(preds), [list(targets)], tokenize='zh'
            ).score
        except Exception as e:
            print(f"BLEU computation error: {e}")
            bleu = 0.0
        
        # ROUGE scores
        rouge_scores = []
        for p, t in valid_pairs:
            try:
                scores = self.rouge_scorer.score(t, p)
                rouge_scores.append({
                    'rouge1': scores['rouge1'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure
                })
            except:
                rouge_scores.append({'rouge1': 0.0, 'rougeL': 0.0})
        
        rouge1 = np.mean([s['rouge1'] for s in rouge_scores])
        rougeL = np.mean([s['rougeL'] for s in rouge_scores])
        
        # Character-level accuracy
        char_correct = 0
        char_total = 0
        for p, t in valid_pairs:
            min_len = min(len(p), len(t))
            char_correct += sum(1 for i in range(min_len) if p[i] == t[i])
            char_total += len(t)
        
        char_accuracy = char_correct / char_total if char_total > 0 else 0.0
        
        # Semantic similarity
        try:
            sample_size = min(50, len(preds))
            sample_indices = np.random.choice(len(preds), sample_size, replace=False)
            sample_preds = [preds[i] for i in sample_indices]
            sample_targets = [targets[i] for i in sample_indices]
            
            with torch.no_grad():
                dummy_eeg = torch.zeros(sample_size, self.semantic_loss.hidden_dim).to(self.device)
                _, semantic_sim = self.semantic_loss(dummy_eeg, sample_preds, sample_targets)
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            semantic_sim = 0.0
        
        # Grammatical score
        try:
            grammatical_loss = self.grammatical_loss(preds, targets)
            grammatical_score = 1.0 - grammatical_loss.item()
        except Exception as e:
            print(f"Grammatical score error: {e}")
            grammatical_score = 0.0
        
        # Fluency score
        try:
            fluency_loss = self.fluency_loss(preds)
            fluency_score = 1.0 - fluency_loss.item()
        except Exception as e:
            print(f"Fluency score error: {e}")
            fluency_score = 0.0
        
        # Average lengths
        avg_pred_len = np.mean([len(p) for p in preds])
        avg_target_len = np.mean([len(t) for t in targets])
        
        return {
            'bleu': bleu,
            'rouge1': rouge1,
            'rougeL': rougeL,
            'semantic_sim': semantic_sim,
            'char_accuracy': char_accuracy,
            'grammatical_score': grammatical_score,
            'fluency_score': fluency_score,
            'avg_pred_len': avg_pred_len,
            'avg_target_len': avg_target_len
        }