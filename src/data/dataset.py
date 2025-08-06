import os
import pickle
from . import utils
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import re
from sklearn.preprocessing import RobustScaler
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class EEGDataset(Dataset):
    """Fixed EEG dataset with safer tokenization and error handling"""
    
    def __init__(self, data_dir, csv_path, tokenizer, max_length=64, eps=1e-6, 
                 max_samples=None, data_augmentation=True, text_preprocessing=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eps = eps
        self.max_samples = max_samples
        self.data_augmentation = data_augmentation
        self.text_preprocessing = text_preprocessing

        # CRITICAL: Store vocabulary size for validation
        self.vocab_size = len(tokenizer.get_vocab())
        logger.info(f"Tokenizer vocabulary size: {self.vocab_size}")

        # Load channel names
        df = pd.read_csv(csv_path)
        self.ch_names = df['label'].to_numpy()

        # Build and validate region indices
        self.region_indices = self._build_region_indices()
        self.region_channel_counts = {
            region: len(indices) for region, indices in self.region_indices.items()
        }
        self._validate_region_indices()

        # CRITICAL: Safe tokenizer setup
        self._setup_tokenizer_safe()

        # Data loading with better error handling
        print("Setting up enhanced data loading...")
        self.data_files = self._get_validated_data_files(data_dir)
        self.sample_indices = self._build_robust_sample_index()
        
        # Text preprocessing patterns
        if self.text_preprocessing:
            self._setup_text_preprocessing()

        # Initialize scalers dictionary
        self.scalers = defaultdict(lambda: RobustScaler())
        
        # Precompute scalers on a subset of data
        self._precompute_scalers()

    def _build_region_indices(self):
        """Build and validate region indices"""
        region_indices = {
            'frontal': [i for i, ch in enumerate(self.ch_names) if ch in utils.frontal_electrodes],
            'temporal': [i for i, ch in enumerate(self.ch_names) if ch in utils.temporal_electrodes],
            'central': [i for i, ch in enumerate(self.ch_names) if ch in utils.central_electrodes],
            'parietal': [i for i, ch in enumerate(self.ch_names) if ch in utils.parietal_electrodes],
        }
        
        # Log region information
        for region, indices in region_indices.items():
            channels = [self.ch_names[i] for i in indices]
            logger.info(f"{region.capitalize()} region: {len(indices)} channels - {channels}")
        
        return region_indices

    def _validate_region_indices(self):
        """Validate that all regions have channels"""
        total_channels = sum(len(indices) for indices in self.region_indices.values())
        logger.info(f"Total channels mapped: {total_channels}/{len(self.ch_names)}")
        
        for region, indices in self.region_indices.items():
            if not indices:
                raise ValueError(f"No channels found for {region} region!")
            elif len(indices) < 3:
                logger.warning(f"Very few channels ({len(indices)}) for {region} region")

    def _setup_tokenizer_safe(self):
        """CRITICAL: Safe tokenizer setup with proper validation"""
        # Ensure padding token is set correctly
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Set pad_token_id to {self.tokenizer.pad_token_id}")

        # CRITICAL: Do NOT add special tokens that could cause indexing issues
        # Instead, use standard tokens or map to existing ones
        logger.info("Using existing tokenizer vocabulary without modifications")
        
        # Validate key token IDs are within bounds
        key_ids = [
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0
        ]
        
        for token_id in key_ids:
            if token_id is not None and token_id >= self.vocab_size:
                raise ValueError(f"Token ID {token_id} >= vocabulary size {self.vocab_size}")
        
        logger.info(f"Tokenizer validation passed. Key IDs: pad={self.tokenizer.pad_token_id}, "
                   f"eos={self.tokenizer.eos_token_id}, bos={self.tokenizer.bos_token_id}")

    def _setup_text_preprocessing(self):
        """Setup conservative text preprocessing patterns"""
        # Conservative Chinese text normalization
        self.text_patterns = [
            # Remove extra whitespace
            (re.compile(r'\s+'), ' '),
            # Remove special characters but keep Chinese and basic punctuation
            (re.compile(r'[^\u4e00-\u9fff\w\s\.,!?;:\-()]'), ''),
            # Normalize simple numbers to a common token (using existing vocab)
            (re.compile(r'\d+'), '数字'),  # "number" in Chinese - likely in vocab
        ]
        
        logger.info("Setup conservative text preprocessing patterns")

    def _get_validated_data_files(self, data_dir):
        """Get and validate data files"""
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) 
                if fname.endswith('.pkl')]
        
        if not files:
            raise ValueError(f"No .pkl files found in {data_dir}")
        
        # Validate file accessibility
        valid_files = []
        for file_path in files:
            try:
                with open(file_path, 'rb') as f:
                    # Try to load first few bytes
                    pickle.load(f)
                valid_files.append(file_path)
            except Exception as e:
                logger.warning(f"Skipping corrupted file {file_path}: {e}")
        
        logger.info(f"Found {len(valid_files)} valid data files out of {len(files)} total")
        return valid_files

    def _build_robust_sample_index(self):
        """Build sample index with better error handling"""
        sample_indices = []
        total_samples = 0
        corrupted_files = 0

        for file_path in self.data_files:
            try:
                with open(file_path, 'rb') as f:
                    loaded = pickle.load(f)
                
                if isinstance(loaded, list):
                    num_samples = len(loaded)
                    for i in range(num_samples):
                        # Validate sample has required fields
                        if self._validate_sample(loaded[i]):
                            sample_indices.append((file_path, i))
                            total_samples += 1
                            if self.max_samples and total_samples >= self.max_samples:
                                break
                else:
                    # Single sample file
                    if self._validate_sample(loaded):
                        sample_indices.append((file_path, 0))
                        total_samples += 1
                        
            except Exception as e:
                logger.warning(f"Error indexing {file_path}: {e}")
                corrupted_files += 1
                continue
            
            if self.max_samples and total_samples >= self.max_samples:
                break

        logger.info(f"Indexed {len(sample_indices)} valid samples from {len(self.data_files)} files")
        if corrupted_files > 0:
            logger.warning(f"Skipped {corrupted_files} corrupted files")
        
        return sample_indices

    def _validate_sample(self, sample):
        """Validate that a sample has required fields and dimensions"""
        if not isinstance(sample, dict):
            return False
        
        required_fields = ['input_features', 'text']
        if not all(field in sample for field in required_fields):
            return False
            
        # Validate EEG data dimensions
        eeg_data = sample['input_features']
        if not isinstance(eeg_data, (list, np.ndarray)):
            return False
            
        # Ensure we have 125 channels as expected
        eeg_array = np.array(eeg_data)
        if len(eeg_array.shape) < 2 or eeg_array.shape[1] != 125:
            logger.warning(f"Invalid EEG shape: {eeg_array.shape} instead of (x, 125)")
            return False
            
        return True

    def _precompute_scalers(self, subset_size=500):
        """Precompute scalers on a subset of data"""
        if not self.sample_indices:
            logger.warning("No samples available for precomputing scalers")
            return

        # Initialize data collectors for each region
        region_data = {region: [] for region in self.region_indices}
        sample_count = min(subset_size, len(self.sample_indices))
        indices = np.random.choice(len(self.sample_indices), sample_count, replace=False)
        
        logger.info(f"Precomputing scalers on {sample_count} samples...")
        
        successful_samples = 0
        for idx in indices:
            file_path, sample_idx = self.sample_indices[idx]
            sample = self.load_single_sample(file_path, sample_idx)
            if sample is None:
                continue
                
            eeg = np.array(sample['input_features'], dtype=np.float32).squeeze()
            
            # Handle shape issues
            if eeg.ndim == 1:
                eeg = eeg.reshape(1, -1)
            elif eeg.ndim > 2:
                eeg = eeg.reshape(eeg.shape[0], -1)
            
            # Clean data
            if not np.isfinite(eeg).all():
                eeg = np.nan_to_num(eeg, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Process each region
            for region_name, indices in self.region_indices.items():
                try:
                    region_eeg = eeg[indices].astype(np.float32)
                    region_data[region_name].append(region_eeg)
                except IndexError:
                    logger.warning(f"Index error for region {region_name}")
                    continue
            
            successful_samples += 1
        
        # Fit scalers on collected data
        for region_name, data_list in region_data.items():
            if not data_list:
                continue
                
            try:
                # Concatenate along time dimension
                all_data = np.concatenate(data_list, axis=1).T
                if all_data.size > 0:
                    self.scalers[region_name].fit(all_data)
                    logger.info(f"Fitted scaler for {region_name} region with {all_data.shape[0]} samples")
            except Exception as e:
                logger.warning(f"Failed to fit scaler for {region_name}: {e}")
        
        logger.info(f"Finished precomputing scalers on {successful_samples} successful samples")

    def _preprocess_text_safe(self, text):
        """CRITICAL: Safe text preprocessing that won't create OOV tokens"""
        if not self.text_preprocessing or not text:
            return text
        
        try:
            # Apply conservative normalization patterns
            for pattern, replacement in self.text_patterns:
                text = pattern.sub(replacement, text)
            
            # Clean up
            text = text.strip()
            
            # Ensure text is not empty
            if not text:
                text = "文本"  # "text" in Chinese - very likely to be in vocab
                
            return text
            
        except Exception as e:
            logger.warning(f"Text preprocessing failed: {e}")
            return "文本"  # Fallback

    def _safe_tokenize(self, text):
        """CRITICAL: Safe tokenization with validation"""
        try:
            # Preprocess text safely
            text = self._preprocess_text_safe(text)
            
            # Tokenize with safety checks
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )
            
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            # CRITICAL: Validate all token IDs are within vocabulary
            max_id = input_ids.max().item()
            min_id = input_ids.min().item()
            
            if max_id >= self.vocab_size:
                logger.error(f"Token ID {max_id} >= vocab_size {self.vocab_size} for text: '{text}'")
                raise ValueError(f"Invalid token ID: {max_id}")
            
            if min_id < 0:
                logger.error(f"Negative token ID {min_id} for text: '{text}'")
                raise ValueError(f"Negative token ID: {min_id}")
            
            # Create decoder input IDs safely
            decoder_start_token_id = (
                self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None
                else self.tokenizer.eos_token_id
            )
            
            # Ensure decoder start token is valid
            if decoder_start_token_id >= self.vocab_size:
                decoder_start_token_id = self.tokenizer.eos_token_id
            
            decoder_input_ids = shift_tokens_right(
                input_ids.unsqueeze(0),
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=decoder_start_token_id
            ).squeeze(0)
            
            # Validate decoder input IDs
            max_decoder_id = decoder_input_ids.max().item()
            if max_decoder_id >= self.vocab_size:
                logger.error(f"Decoder token ID {max_decoder_id} >= vocab_size")
                # Fix by replacing with pad token
                decoder_input_ids[decoder_input_ids >= self.vocab_size] = self.tokenizer.pad_token_id
            
            # Create labels
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'decoder_input_ids': decoder_input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
            }
            
        except Exception as e:
            logger.error(f"Tokenization failed for text '{text}': {e}")
            return self._create_fallback_tokenization()

    def _create_fallback_tokenization(self):
        """Create safe fallback tokenization"""
        try:
            # Use a very simple, safe text
            fallback_text = "数据"  # "data" in Chinese
            
            # Simple tokenization
            tokens = self.tokenizer.tokenize(fallback_text)
            if not tokens:
                # If tokenization fails, create minimal valid sequence
                input_ids = torch.tensor([
                    self.tokenizer.eos_token_id if self.tokenizer.eos_token_id < self.vocab_size else 0
                ] + [self.tokenizer.pad_token_id] * (self.max_length - 1))
            else:
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                # Ensure all IDs are valid
                token_ids = [tid for tid in token_ids if 0 <= tid < self.vocab_size]
                if not token_ids:
                    token_ids = [self.tokenizer.eos_token_id if self.tokenizer.eos_token_id < self.vocab_size else 0]
                
                # Pad to max length
                input_ids = torch.tensor(token_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(token_ids)))
                input_ids = input_ids[:self.max_length]  # Truncate if needed
            
            # Create attention mask
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            
            # Create decoder input
            decoder_input_ids = torch.roll(input_ids, 1)
            decoder_input_ids[0] = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id < self.vocab_size else 0
            
            # Create labels
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'decoder_input_ids': decoder_input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
            }
            
        except Exception as e:
            logger.error(f"Even fallback tokenization failed: {e}")
            # Last resort: create minimal tensor
            return {
                'decoder_input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.full((self.max_length,), -100, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            }

    def _advanced_eeg_preprocessing(self, eeg_data):
        """Advanced EEG preprocessing with robust normalization"""
        try:
            eeg = np.array(eeg_data, dtype=np.float32).squeeze()
            
            # Handle shape issues
            if eeg.ndim == 1:
                eeg = eeg.reshape(1, -1)
            elif eeg.ndim > 2:
                eeg = eeg.reshape(eeg.shape[0], -1)
            
            # Clean data
            if not np.isfinite(eeg).all():
                logger.debug("Cleaning non-finite values in EEG data")
                eeg = np.nan_to_num(eeg, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Clip extreme values (beyond 6 standard deviations)
            std_thresh = 6.0
            mean_val = np.mean(eeg)
            std_val = np.std(eeg)
            if std_val > 0:
                eeg = np.clip(eeg, mean_val - std_thresh * std_val, mean_val + std_thresh * std_val)
            
            # Process each region
            processed_regions = []
            for region_name in ['frontal', 'temporal', 'central', 'parietal']:
                indices = self.region_indices[region_name]
                
                # Extract region data safely
                try:
                    region_data = eeg[indices].astype(np.float32)
                except IndexError:
                    logger.warning(f"Index error for region {region_name}, using zeros")
                    region_data = np.zeros((len(indices), eeg.shape[1]), dtype=np.float32)
                
                # Apply scaler if available
                if region_data.size > 0 and hasattr(self.scalers[region_name], 'scale_'):
                    try:
                        # Transform data (scaler expects [samples, features])
                        region_data = self.scalers[region_name].transform(region_data.T).T
                    except Exception as e:
                        logger.warning(f"Scaling failed for {region_name}: {e}")
                    
                    # Apply data augmentation with reduced probability
                    if self.data_augmentation and np.random.rand() < 0.1:  # Reduced probability
                        region_data = self._apply_eeg_augmentation(region_data)
                
                processed_regions.append(region_data)
            
            return processed_regions
            
        except Exception as e:
            logger.error(f"EEG preprocessing failed: {e}")
            # Return correctly dimensioned dummy data
            return [np.zeros((len(self.region_indices[region]), 1651), dtype=np.float32)
                   for region in ['frontal', 'temporal', 'central', 'parietal']]

    def _apply_eeg_augmentation(self, region_data):
        """Apply minimal data augmentation to EEG data"""
        try:
            augmented = region_data.copy()
            
            # Very light noise addition (2% of std)
            if np.random.rand() < 0.5:
                noise_std = max(np.std(augmented) * 0.02, 1e-6)
                noise = np.random.normal(0, noise_std, augmented.shape)
                augmented += noise
            
            # Very small amplitude scaling (±5%)
            if np.random.rand() < 0.3:
                scale_factor = np.random.uniform(0.95, 1.05)
                augmented *= scale_factor
            
            return augmented
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return region_data

    def load_single_sample(self, file_path, sample_idx):
        """Load single sample with better error handling"""
        try:
            with open(file_path, 'rb') as f:
                loaded = pickle.load(f)
            
            if isinstance(loaded, list):
                if sample_idx < len(loaded):
                    sample = loaded[sample_idx]
                    return sample if self._validate_sample(sample) else None
                else:
                    return None
            else:
                return loaded if (sample_idx == 0 and self._validate_sample(loaded)) else None
                
        except Exception as e:
            logger.error(f"Error loading sample from {file_path}: {e}")
            return None

    def _create_fallback_sample(self):
        """Create fallback sample for corrupted data with proper dimensions"""
        eeg_regions = [
            np.zeros((len(self.region_indices[region]), 1651), dtype=np.float32)
            for region in ['frontal', 'temporal', 'central', 'parietal']
        ]
        
        fallback_tokenization = self._create_fallback_tokenization()
        
        return {
            'eeg': eeg_regions,
            **fallback_tokenization
        }

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        """Enhanced getitem with comprehensive error handling and safe tokenization"""
        if idx >= len(self.sample_indices):
            logger.error(f"Index {idx} out of range")
            return self._create_fallback_sample()

        file_path, sample_idx = self.sample_indices[idx]
        sample = self.load_single_sample(file_path, sample_idx)

        if sample is None:
            logger.warning(f"Failed to load sample {idx}, using fallback")
            return self._create_fallback_sample()

        try:
            # Process EEG data
            eeg_regions = self._advanced_eeg_preprocessing(sample['input_features'])
            
            # Process text safely
            text = sample.get('text', '')
            if not text or len(text.strip()) == 0:
                text = "无内容"  # "no content" in Chinese
            
            # CRITICAL: Safe tokenization
            tokenization_result = self._safe_tokenize(text)
            
            return {
                'eeg': eeg_regions,
                **tokenization_result
            }
            
        except Exception as e:
            logger.error(f"Sample processing failed for index {idx}: {e}")
            return self._create_fallback_sample()

    def get_sample_stats(self):
        """Get dataset statistics for analysis"""
        if not hasattr(self, '_stats'):
            logger.info("Computing dataset statistics...")
            
            text_lengths = []
            eeg_shapes = []
            valid_samples = 0
            
            # Sample a subset for statistics
            sample_size = min(1000, len(self.sample_indices))
            indices = np.random.choice(len(self.sample_indices), sample_size, replace=False)
            
            for idx in indices:
                try:
                    file_path, sample_idx = self.sample_indices[idx]
                    sample = self.load_single_sample(file_path, sample_idx)
                    
                    if sample and self._validate_sample(sample):
                        text = sample.get('text', '')
                        if text:
                            # Use tokenized length instead of raw text length
                            try:
                                tokens = self.tokenizer.tokenize(text)
                                text_lengths.append(len(tokens))
                            except:
                                text_lengths.append(len(text.split()))
                        
                        eeg_data = sample.get('input_features')
                        if eeg_data is not None:
                            eeg_shapes.append(np.array(eeg_data).shape)
                        
                        valid_samples += 1
                        
                except Exception:
                    continue
            
            self._stats = {
                'total_samples': len(self.sample_indices),
                'valid_samples': valid_samples,
                'avg_text_length': np.mean(text_lengths) if text_lengths else 0,
                'text_length_std': np.std(text_lengths) if text_lengths else 0,
                'unique_eeg_shapes': len(set(tuple(shape) for shape in eeg_shapes)),
                'most_common_eeg_shape': max(set(tuple(shape) for shape in eeg_shapes), 
                                           key=lambda x: [tuple(shape) for shape in eeg_shapes].count(x)) if eeg_shapes else None,
                'region_channel_counts': self.region_channel_counts
            }
            
            logger.info(f"Dataset stats computed: {self._stats}")
        
        return self._stats