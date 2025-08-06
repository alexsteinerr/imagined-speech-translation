import os
import pickle
import utils
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import re
from sklearn.preprocessing import RobustScaler
import logging
from collections import defaultdict
from functools import lru_cache
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib

logger = logging.getLogger(__name__)

class EEGDataset(Dataset):
    """
    Highly optimized EEG dataset with enhanced alignment validation, 
    efficient file-based loading, and comprehensive error handling.
    No memory preloading - optimized for large datasets.
    """
    
    def __init__(self, data_dir, csv_path, tokenizer, max_length=64, eps=1e-6, 
                 max_samples=None, data_augmentation=False, text_preprocessing=True,
                 cache_size=2000, num_workers=0, lazy_scaling=True, 
                 validation_samples=1000):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eps = eps
        self.max_samples = max_samples
        self.data_augmentation = data_augmentation
        self.text_preprocessing = text_preprocessing
        self.cache_size = cache_size
        self.num_workers = num_workers
        self.lazy_scaling = lazy_scaling
        self.validation_samples = validation_samples
        
        # Thread-safe caching
        self._sample_cache = {}
        self._tokenization_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Pre-compute vocabulary size once
        self.vocab_size = len(tokenizer.get_vocab())
        logger.info(f"Tokenizer vocabulary size: {self.vocab_size}")
        
        # Load channel names efficiently
        df = pd.read_csv(csv_path)
        self.ch_names = df['label'].values  # More efficient than to_numpy()
        
        # Build region indices with vectorized operations
        self.region_indices = self._build_region_indices_vectorized()
        self.region_channel_counts = {
            region: len(indices) for region, indices in self.region_indices.items()
        }
        self._validate_region_indices()
        
        # Setup tokenizer safely
        self._setup_tokenizer_safe()
        
        # Setup text preprocessing patterns if enabled
        if self.text_preprocessing:
            self._setup_text_preprocessing_patterns()
        
        # Enhanced file validation and indexing
        logger.info("Setting up enhanced data loading with alignment checking...")
        self.data_files = self._get_validated_data_files_parallel(data_dir)
        self.sample_indices = self._build_robust_sample_index_parallel()
        
        # Initialize scalers (lazy or immediate)
        if self.lazy_scaling:
            self.scalers = None
        else:
            self.scalers = defaultdict(lambda: RobustScaler())
            self._precompute_scalers_parallel()
        
        # Pre-allocate common arrays
        self._preallocate_common_arrays()
        
        logger.info(f"Dataset initialized with {len(self.sample_indices)} samples")

    def _build_region_indices_vectorized(self):
        """Vectorized region index building for better performance"""
        electrode_sets = {
            'frontal': frozenset(utils.frontal_electrodes),
            'temporal': frozenset(utils.temporal_electrodes), 
            'central': frozenset(utils.central_electrodes),
            'parietal': frozenset(utils.parietal_electrodes),
        }
        
        # Vectorized approach with frozenset for O(1) lookup
        indices = {}
        for region, electrode_set in electrode_sets.items():
            mask = np.isin(self.ch_names, list(electrode_set))
            indices[region] = np.where(mask)[0]
            
            # Log region information
            channels = self.ch_names[indices[region]]
            logger.info(f"{region.capitalize()} region: {len(indices[region])} channels - "
                       f"{list(channels[:5])}{'...' if len(channels) > 5 else ''}")
        
        return indices

    def _validate_region_indices(self):
        """Enhanced region validation with warnings"""
        total_channels = sum(len(indices) for indices in self.region_indices.values())
        logger.info(f"Total channels mapped: {total_channels}/{len(self.ch_names)}")
        
        for region, indices in self.region_indices.items():
            if not indices.size:
                raise ValueError(f"No channels found for {region} region!")
            elif len(indices) < 2:
                logger.warning(f"Very few channels ({len(indices)}) for {region} region")

    def _setup_tokenizer_safe(self):
        """Safe tokenizer setup with comprehensive validation"""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Set pad_token_id to {self.tokenizer.pad_token_id}")

        # Validate key token IDs
        key_ids = {
            'pad': self.tokenizer.pad_token_id,
            'eos': self.tokenizer.eos_token_id,
            'bos': self.tokenizer.bos_token_id
        }
        
        for name, token_id in key_ids.items():
            if token_id is not None and token_id >= self.vocab_size:
                raise ValueError(f"{name.upper()} token ID {token_id} >= vocabulary size {self.vocab_size}")
        
        logger.info(f"Tokenizer validation passed. Key IDs: {key_ids}")

    def _setup_text_preprocessing_patterns(self):
        """Setup optimized text preprocessing patterns"""
        # Compile regex patterns once for better performance
        self.text_patterns = [
            (re.compile(r'\s+'), ' '),  # Normalize whitespace
            (re.compile(r'[^\u4e00-\u9fff\w\s\.,!?;:\-()，。！？；：]'), ''),  # Keep Chinese + basic punct
            (re.compile(r'[""''`´]'), '"'),  # Normalize quotes
        ]
        
        # Fallback texts for different scenarios
        self.fallback_texts = {
            'empty': "无内容",
            'short': "短文本", 
            'invalid': "无效文本",
            'error': "处理错误"
        }
        
        logger.info("Setup optimized text preprocessing patterns")

    def _get_validated_data_files_parallel(self, data_dir):
        """Parallel file validation for faster startup"""
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        pkl_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) 
                    if fname.endswith('.pkl')]
        
        if not pkl_files:
            raise ValueError(f"No .pkl files found in {data_dir}")
        
        def validate_file(file_path):
            """Quick file validation without full loading"""
            try:
                # Check file accessibility and basic pickle format
                with open(file_path, 'rb') as f:
                    header = f.read(10)
                    if not header or header[:2] not in [b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05']:
                        return None
                
                # Quick structure validation
                with open(file_path, 'rb') as f:
                    try:
                        test_data = pickle.load(f)
                        if isinstance(test_data, list):
                            if test_data and self._quick_validate_sample(test_data[0]):
                                return file_path
                        elif self._quick_validate_sample(test_data):
                            return file_path
                    except:
                        pass
                        
            except Exception as e:
                logger.debug(f"File validation failed for {file_path}: {e}")
            
            return None
        
        # Parallel validation
        valid_files = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(validate_file, file_path): file_path 
                      for file_path in sorted(pkl_files)}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    valid_files.append(result)
        
        logger.info(f"Validated {len(valid_files)} files out of {len(pkl_files)} total")
        return sorted(valid_files)  # Sort for reproducibility

    def _quick_validate_sample(self, sample):
        """Optimized sample validation"""
        return (isinstance(sample, dict) and
                'input_features' in sample and
                'text' in sample and
                len(sample.get('text', '').strip()) >= 2)

    def _build_robust_sample_index_parallel(self):
        """Parallel sample indexing with enhanced validation"""
        def index_file(file_path):
            """Index samples in a single file"""
            file_indices = []
            try:
                with open(file_path, 'rb') as f:
                    loaded = pickle.load(f)
                
                if isinstance(loaded, list):
                    for i, sample in enumerate(loaded):
                        if self._validate_sample_alignment(sample):
                            file_indices.append((file_path, i))
                else:
                    if self._validate_sample_alignment(loaded):
                        file_indices.append((file_path, 0))
                        
            except Exception as e:
                logger.debug(f"Error indexing {file_path}: {e}")
            
            return file_indices
        
        # Parallel indexing
        all_indices = []
        total_samples = 0
        alignment_issues = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(index_file, file_path): file_path 
                      for file_path in self.data_files}
            
            for future in as_completed(futures):
                file_indices = future.result()
                all_indices.extend(file_indices)
                
                if self.max_samples and len(all_indices) >= self.max_samples:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
        
        # Truncate to max_samples if specified
        if self.max_samples:
            all_indices = all_indices[:self.max_samples]
        
        logger.info(f"Indexed {len(all_indices)} valid samples")
        return all_indices

    def _validate_sample_alignment(self, sample):
        """Comprehensive sample validation including EEG-text alignment"""
        if not isinstance(sample, dict):
            return False
        
        # Check required fields
        if not all(field in sample for field in ['input_features', 'text']):
            return False
        
        # Validate EEG data
        try:
            eeg_data = sample['input_features']
            if not isinstance(eeg_data, (list, np.ndarray)):
                return False
            
            eeg_array = np.array(eeg_data)
            
            # Check dimensions (expecting channels x time)
            if eeg_array.ndim < 2:
                return False
            
            # Check for finite values
            if not np.isfinite(eeg_array).all():
                return False
            
            # Check for reasonable variance (not all zeros)
            if np.std(eeg_array) < 1e-6:
                return False
                
        except:
            return False
        
        # Validate text
        text = sample.get('text', '').strip()
        if not text or len(text) < 2:
            return False
        
        # Check for meaningful content
        meaningful_chars = sum(1 for c in text if c.isalnum() or ord(c) > 127)
        if meaningful_chars < 2:
            return False
        
        return True

    def _preallocate_common_arrays(self):
        """Pre-allocate commonly used arrays for efficiency"""
        self._percentiles = np.array([25, 75])
        self._max_channels = max(len(indices) for indices in self.region_indices.values())
        self._region_names = ['frontal', 'temporal', 'central', 'parietal']
        
        # Pre-compute fallback shapes
        self._fallback_shapes = {
            region: (len(self.region_indices[region]), 1651)
            for region in self._region_names
        }

    @lru_cache(maxsize=1000)
    def _get_text_hash(self, text):
        """Create hash for text caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _preprocess_text_optimized(self, text):
        """Optimized text preprocessing with caching"""
        if not self.text_preprocessing or not text:
            return self.fallback_texts['empty']
        
        text_hash = self._get_text_hash(text)
        
        # Check cache first
        with self._cache_lock:
            if text_hash in self._tokenization_cache:
                self._cache_hits += 1
                return self._tokenization_cache[text_hash]
            self._cache_misses += 1
        
        try:
            original_text = text
            
            # Apply preprocessing patterns
            for pattern, replacement in self.text_patterns:
                text = pattern.sub(replacement, text)
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            # Ensure meaningful content
            if not text or len(text.strip()) < 1:
                meaningful_chars = ''.join(c for c in original_text 
                                         if c.isalnum() or ord(c) > 127)
                text = meaningful_chars[:50] if len(meaningful_chars) >= 2 else self.fallback_texts['empty']
            
            result = text.strip()
            
            # Cache result
            with self._cache_lock:
                if len(self._tokenization_cache) >= self.cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._tokenization_cache))
                    del self._tokenization_cache[oldest_key]
                self._tokenization_cache[text_hash] = result
            
            return result
            
        except Exception as e:
            logger.debug(f"Text preprocessing failed: {e}")
            return self.fallback_texts['error']

    def _safe_tokenize_enhanced(self, text):
        """Enhanced tokenization with comprehensive error handling"""
        try:
            # Preprocess text
            text = self._preprocess_text_optimized(text)
            
            # Ensure minimum length
            if len(text) < 2:
                text = self.fallback_texts['short']
            
            # Test tokenization capability
            try:
                test_tokens = self.tokenizer.tokenize(text)
                if not test_tokens:
                    text = self.fallback_texts['invalid']
            except:
                text = self.fallback_texts['error']
            
            # Full tokenization
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
            
            # Validate token IDs
            max_id = input_ids.max().item()
            if max_id >= self.vocab_size:
                logger.debug(f"Invalid token ID {max_id} for text: '{text}'")
                return self._create_emergency_fallback_tokenization()
            
            # Create decoder input IDs
            decoder_start_token_id = (
                self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None
                else self.tokenizer.eos_token_id
            )
            
            decoder_input_ids = shift_tokens_right(
                input_ids.unsqueeze(0),
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=decoder_start_token_id
            ).squeeze(0)
            
            # Validate decoder IDs
            if decoder_input_ids.max().item() >= self.vocab_size:
                decoder_input_ids[decoder_input_ids >= self.vocab_size] = self.tokenizer.pad_token_id
            
            # Create labels
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'decoder_input_ids': decoder_input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }
            
        except Exception as e:
            logger.debug(f"Enhanced tokenization failed: {e}")
            return self._create_emergency_fallback_tokenization()

    def _create_emergency_fallback_tokenization(self):
        """Create guaranteed valid tokenization"""
        if not hasattr(self, '_cached_fallback'):
            try:
                safe_text = self.fallback_texts['empty']  # "数据"
                
                tokens = self.tokenizer.encode(
                    safe_text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length'
                )
                
                input_ids = torch.tensor(tokens, dtype=torch.long)
                
                # Ensure all IDs are valid
                if input_ids.max().item() >= self.vocab_size:
                    input_ids = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                    input_ids[0] = min(self.tokenizer.eos_token_id, self.vocab_size - 1)
                
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                decoder_input_ids = torch.roll(input_ids, 1)
                decoder_input_ids[0] = min(self.tokenizer.eos_token_id, self.vocab_size - 1)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                self._cached_fallback = {
                    'decoder_input_ids': decoder_input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask
                }
                
            except Exception as e:
                logger.error(f"Emergency fallback creation failed: {e}")
                # Absolute last resort
                self._cached_fallback = {
                    'decoder_input_ids': torch.zeros(self.max_length, dtype=torch.long),
                    'labels': torch.full((self.max_length,), -100, dtype=torch.long),
                    'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
                }
        
        # Return cloned tensors to avoid shared state
        return {k: v.clone() for k, v in self._cached_fallback.items()}

    def _get_or_create_scalers(self):
        """Lazy scaler initialization"""
        if self.scalers is None:
            logger.info("Computing scalers on-demand...")
            self.scalers = defaultdict(lambda: RobustScaler())
            self._precompute_scalers_parallel()
        return self.scalers

    def _precompute_scalers_parallel(self):
        """Parallel scaler computation for efficiency"""
        if not self.sample_indices:
            return
        
        # Sample subset for scaler computation
        sample_count = min(self.validation_samples, len(self.sample_indices))
        indices = np.random.choice(len(self.sample_indices), sample_count, replace=False)
        
        def load_eeg_for_scaling(idx):
            """Load EEG data for a single sample"""
            try:
                file_path, sample_idx = self.sample_indices[idx]
                sample = self._load_sample_cached(file_path, sample_idx)
                if sample:
                    eeg = np.array(sample['input_features'], dtype=np.float32)
                    eeg = self._normalize_eeg_shape(eeg)
                    eeg = np.nan_to_num(eeg, nan=0.0, posinf=1.0, neginf=-1.0)
                    return eeg
            except:
                pass
            return None
        
        # Parallel loading
        eeg_samples = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(load_eeg_for_scaling, idx): idx for idx in indices}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    eeg_samples.append(result)
        
        # Process regions and fit scalers
        region_data = {region: [] for region in self.region_indices}
        
        for eeg in eeg_samples:
            for region_name, region_indices in self.region_indices.items():
                try:
                    region_eeg = eeg[region_indices]
                    region_data[region_name].append(region_eeg)
                except IndexError:
                    continue
        
        # Fit scalers efficiently
        for region_name, data_list in region_data.items():
            if data_list:
                try:
                    # Use vstack for better memory efficiency
                    all_data = np.vstack([data.T for data in data_list])
                    if all_data.size > 0:
                        self.scalers[region_name].fit(all_data)
                        logger.info(f"Fitted scaler for {region_name} region with {all_data.shape[0]} samples")
                except Exception as e:
                    logger.warning(f"Failed to fit scaler for {region_name}: {e}")

    def _normalize_eeg_shape(self, eeg):
        """Efficiently normalize EEG array shape"""
        if eeg.ndim == 1:
            return eeg.reshape(1, -1)
        elif eeg.ndim > 2:
            return eeg.reshape(eeg.shape[0], -1)
        return eeg

    def _load_sample_cached(self, file_path, sample_idx):
        """Cached sample loading with LRU eviction"""
        cache_key = (file_path, sample_idx)
        
        with self._cache_lock:
            if cache_key in self._sample_cache:
                # Move to end (LRU)
                value = self._sample_cache.pop(cache_key)
                self._sample_cache[cache_key] = value
                return value
            
            # Manage cache size
            if len(self._sample_cache) >= self.cache_size:
                # Remove least recently used (first item)
                self._sample_cache.pop(next(iter(self._sample_cache)))
        
        # Load sample
        sample = self._load_sample_direct(file_path, sample_idx)
        
        # Cache result
        with self._cache_lock:
            self._sample_cache[cache_key] = sample
            
        return sample

    def _load_sample_direct(self, file_path, sample_idx):
        """Direct sample loading without caching"""
        try:
            with open(file_path, 'rb') as f:
                loaded = pickle.load(f)
            
            if isinstance(loaded, list):
                if sample_idx < len(loaded):
                    sample = loaded[sample_idx]
                    return sample if self._validate_sample_alignment(sample) else None
            else:
                if sample_idx == 0:
                    return loaded if self._validate_sample_alignment(loaded) else None
            
        except Exception as e:
            logger.debug(f"Error loading sample from {file_path}: {e}")
        
        return None

    def _enhanced_eeg_preprocessing(self, eeg_data):
        """Optimized EEG preprocessing with vectorized operations"""
        try:
            eeg = np.array(eeg_data, dtype=np.float32)
            eeg = self._normalize_eeg_shape(eeg)
            eeg = np.nan_to_num(eeg, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Vectorized outlier clipping
            if eeg.shape[0] > 0 and eeg.shape[1] > 1:
                q1, q3 = np.percentile(eeg, self._percentiles, axis=1, keepdims=True)
                iqr = q3 - q1
                valid_mask = iqr.flatten() > 0
                
                if np.any(valid_mask):
                    lower_bounds = q1 - 3 * iqr
                    upper_bounds = q3 + 3 * iqr
                    eeg = np.clip(eeg, lower_bounds, upper_bounds)
            
            # Process regions efficiently
            processed_regions = []
            scalers = self._get_or_create_scalers() if self.lazy_scaling else self.scalers
            
            for region_name in self._region_names:
                indices = self.region_indices[region_name]
                try:
                    if len(indices) == 0:
                        region_data = np.zeros(self._fallback_shapes[region_name], dtype=np.float32)
                    else:
                        region_data = eeg[indices]
                    
                    # Apply scaling
                    if (scalers and region_name in scalers and 
                        hasattr(scalers[region_name], 'scale_') and 
                        region_data.size > 0 and region_data.shape[1] > 1):
                        try:
                            region_data = scalers[region_name].transform(region_data.T).T
                        except:
                            pass
                    
                    # Light augmentation if enabled
                    if self.data_augmentation and np.random.rand() < 0.15:
                        region_data = self._apply_light_augmentation_vectorized(region_data)
                    
                    processed_regions.append(region_data.astype(np.float32))
                    
                except:
                    fallback_data = np.zeros(self._fallback_shapes[region_name], dtype=np.float32)
                    processed_regions.append(fallback_data)
            
            return processed_regions
            
        except Exception as e:
            logger.debug(f"EEG preprocessing failed: {e}")
            return [np.zeros(self._fallback_shapes[region], dtype=np.float32)
                   for region in self._region_names]

    def _apply_light_augmentation_vectorized(self, region_data):
        """Vectorized light augmentation"""
        try:
            augmented = region_data.copy()
            
            # Vectorized noise addition (1% of std)
            if np.random.rand() < 0.7:
                noise_std = max(np.std(augmented) * 0.01, 1e-8)
                noise = np.random.normal(0, noise_std, augmented.shape).astype(np.float32)
                augmented += noise
            
            # Vectorized scaling (±2%)
            if np.random.rand() < 0.3:
                scale_factor = np.random.uniform(0.98, 1.02)
                augmented *= scale_factor
            
            # Slight time shift (±1 sample)
            if np.random.rand() < 0.2 and augmented.shape[1] > 2:
                shift = np.random.randint(-1, 2)
                if shift != 0:
                    augmented = np.roll(augmented, shift, axis=1)
            
            return augmented
        except:
            return region_data

    def _create_fallback_sample(self):
        """Create optimized fallback sample"""
        if not hasattr(self, '_cached_fallback_sample'):
            eeg_regions = []
            for region in self._region_names:
                shape = self._fallback_shapes[region]
                # Minimal random variation
                base_signal = np.random.randn(*shape).astype(np.float32) * 0.01
                eeg_regions.append(base_signal)
            
            fallback_tokenization = self._create_emergency_fallback_tokenization()
            
            self._cached_fallback_sample = {
                'eeg_shapes': [region.shape for region in eeg_regions],
                'tokenization': fallback_tokenization
            }
        
        # Create fresh tensors for each call
        eeg_tensors = [torch.randn(shape, dtype=torch.float32) * 0.01 
                      for shape in self._cached_fallback_sample['eeg_shapes']]
        
        return {
            'eeg': eeg_tensors,
            **{k: v.clone() for k, v in self._cached_fallback_sample['tokenization'].items()}
        }

    def clear_cache(self):
        """Clear all caches and force garbage collection"""
        with self._cache_lock:
            self._sample_cache.clear()
            self._tokenization_cache.clear()
        
        # Clear LRU cache
        self._get_text_hash.cache_clear()
        
        # Log cache statistics
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        logger.info(f"Cache cleared. Hit rate: {hit_rate:.2%} ({self._cache_hits}/{total_requests})")
        
        # Reset counters
        self._cache_hits = 0
        self._cache_misses = 0
        
        gc.collect()

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        """Optimized item retrieval"""
        if idx >= len(self.sample_indices):
            logger.debug(f"Index {idx} out of range")
            return self._create_fallback_sample()

        file_path, sample_idx = self.sample_indices[idx]
        sample = self._load_sample_cached(file_path, sample_idx)

        if sample is None:
            return self._create_fallback_sample()

        try:
            # Process EEG with enhanced preprocessing
            eeg_regions = self._enhanced_eeg_preprocessing(sample['input_features'])
            
            # Process text safely
            text = sample.get('text', '').strip()
            if not text:
                text = self.fallback_texts['empty']
            
            tokenization_result = self._safe_tokenize_enhanced(text)
            
            # Convert to tensors efficiently
            eeg_tensors = [torch.from_numpy(region) for region in eeg_regions]
            
            return {
                'eeg': eeg_tensors,
                **tokenization_result
            }
            
        except Exception as e:
            logger.debug(f"Sample processing failed for index {idx}: {e}")
            return self._create_fallback_sample()

    def get_sample_stats(self):
        """Optimized dataset statistics computation"""
        if hasattr(self, '_stats'):
            return self._stats
            
        logger.info("Computing enhanced dataset statistics...")
        
        # Initialize collectors
        text_lengths = []
        token_lengths = []
        eeg_shapes = []
        eeg_diversity_stats = []
        valid_samples = 0
        
        # Sample efficiently
        sample_size = min(500, len(self.sample_indices))
        indices = np.random.choice(len(self.sample_indices), sample_size, replace=False)
        
        def analyze_sample(idx):
            """Analyze a single sample for statistics"""
            try:
                file_path, sample_idx = self.sample_indices[idx]
                sample = self._load_sample_cached(file_path, sample_idx)
                
                if sample and self._validate_sample_alignment(sample):
                    stats = {}
                    
                    # Text analysis
                    text = sample.get('text', '').strip()
                    if text:
                        stats['text_length'] = len(text)
                        try:
                            tokens = self.tokenizer.tokenize(text)
                            stats['token_length'] = len(tokens)
                        except:
                            stats['token_length'] = len(text.split())
                    
                    # EEG analysis
                    eeg_data = sample.get('input_features')
                    if eeg_data is not None:
                        eeg_array = np.array(eeg_data)
                        stats['eeg_shape'] = eeg_array.shape
                        stats['eeg_diversity'] = {
                            'mean': float(np.mean(eeg_array)),
                            'std': float(np.std(eeg_array)),
                            'range': float(np.ptp(eeg_array))
                        }
                    
                    return stats
                    
            except Exception as e:
                logger.debug(f"Error analyzing sample {idx}: {e}")
            
            return None
        
        # Parallel analysis
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(analyze_sample, indices))
        
        # Process results
        for result in results:
            if result:
                if 'text_length' in result:
                    text_lengths.append(result['text_length'])
                if 'token_length' in result:
                    token_lengths.append(result['token_length'])
                if 'eeg_shape' in result:
                    eeg_shapes.append(result['eeg_shape'])
                if 'eeg_diversity' in result:
                    eeg_diversity_stats.append(result['eeg_diversity'])
                valid_samples += 1
        
        # Compute diversity metrics
        eeg_mean_diversity = (np.std([stat['mean'] for stat in eeg_diversity_stats]) 
                             if eeg_diversity_stats else 0)
        eeg_std_diversity = (np.std([stat['std'] for stat in eeg_diversity_stats]) 
                            if eeg_diversity_stats else 0)
        
        # Get text diversity (approximate)
        text_diversity = len(set(str(length) for length in text_lengths[:50]))  # Rough estimate
        
        self._stats = {
            'total_samples': len(self.sample_indices),
            'valid_samples': valid_samples,
            'avg_text_length': float(np.mean(text_lengths)) if text_lengths else 0,
            'text_length_std': float(np.std(text_lengths)) if text_lengths else 0,
            'avg_token_length': float(np.mean(token_lengths)) if token_lengths else 0,
            'token_length_std': float(np.std(token_lengths)) if token_lengths else 0,
            'unique_eeg_shapes': len(set(tuple(shape) for shape in eeg_shapes)),
            'most_common_eeg_shape': (max(set(tuple(shape) for shape in eeg_shapes), 
                                        key=lambda x: [tuple(shape) for shape in eeg_shapes].count(x)) 
                                    if eeg_shapes else None),
            'eeg_mean_diversity': float(eeg_mean_diversity),
            'eeg_std_diversity': float(eeg_std_diversity),
            'region_channel_counts': self.region_channel_counts,
            'text_diversity_estimate': text_diversity,
            'cache_enabled': True,
            'lazy_scaling': self.lazy_scaling,
            'data_augmentation': self.data_augmentation
        }
        
        # Log statistics
        logger.info(f"Dataset statistics computed:")
        logger.info(f"  Valid samples: {valid_samples}/{len(self.sample_indices)}")
        logger.info(f"  Avg text length: {self._stats['avg_text_length']:.1f} chars")
        logger.info(f"  Avg token length: {self._stats['avg_token_length']:.1f} tokens")
        logger.info(f"  EEG mean diversity: {self._stats['eeg_mean_diversity']:.6f}")
        logger.info(f"  EEG std diversity: {self._stats['eeg_std_diversity']:.6f}")
        
        # Warnings for low diversity
        if self._stats['eeg_mean_diversity'] < 0.001:
            logger.warning("WARNING: Very low EEG mean diversity detected!")
        if self._stats['eeg_std_diversity'] < 0.001:
            logger.warning("WARNING: Very low EEG std diversity detected!")
        
        return self._stats

    def analyze_sample_alignment(self, n_samples=20):
        """Analyze EEG-text alignment in samples for debugging"""
        logger.info("=== SAMPLE ALIGNMENT ANALYSIS ===")
        
        alignment_results = {
            'eeg_diversity': [],
            'text_lengths': [],
            'potential_issues': []
        }
        
        sample_count = min(n_samples, len(self.sample_indices))
        
        for i in range(sample_count):
            try:
                file_path, sample_idx = self.sample_indices[i]
                sample = self._load_sample_cached(file_path, sample_idx)
                
                if sample:
                    # Analyze EEG
                    eeg_data = np.array(sample['input_features'])
                    eeg_diversity = float(np.std(eeg_data))
                    alignment_results['eeg_diversity'].append(eeg_diversity)
                    
                    # Analyze text
                    text = sample.get('text', '').strip()
                    text_length = len(text)
                    alignment_results['text_lengths'].append(text_length)
                    
                    logger.info(f"Sample {i}: EEG diversity={eeg_diversity:.6f}, "
                              f"Text length={text_length}, Text='{text[:50]}{'...' if len(text) > 50 else ''}'")
                    
                    # Flag potential issues
                    if eeg_diversity < 0.001:
                        alignment_results['potential_issues'].append(f"Sample {i}: Very low EEG diversity")
                    if text_length < 3:
                        alignment_results['potential_issues'].append(f"Sample {i}: Very short text")
                        
            except Exception as e:
                logger.warning(f"Error analyzing sample {i}: {e}")
        
        # Summary analysis
        if alignment_results['eeg_diversity']:
            avg_eeg_diversity = np.mean(alignment_results['eeg_diversity'])
            logger.info(f"Average EEG diversity: {avg_eeg_diversity:.6f}")
            
            if avg_eeg_diversity < 0.01:
                logger.warning("WARNING: Overall low EEG diversity detected!")
        
        if alignment_results['text_lengths']:
            avg_text_length = np.mean(alignment_results['text_lengths'])
            logger.info(f"Average text length: {avg_text_length:.1f} characters")
        
        if alignment_results['potential_issues']:
            logger.warning("Potential alignment issues found:")
            for issue in alignment_results['potential_issues']:
                logger.warning(f"  {issue}")
        else:
            logger.info("No obvious alignment issues detected")
        
        return alignment_results

    def get_cache_info(self):
        """Get cache performance information"""
        with self._cache_lock:
            sample_cache_size = len(self._sample_cache)
            tokenization_cache_size = len(self._tokenization_cache)
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'sample_cache_size': sample_cache_size,
            'tokenization_cache_size': tokenization_cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

    def optimize_cache_size(self, target_hit_rate=0.8):
        """Dynamically adjust cache size based on performance"""
        cache_info = self.get_cache_info()
        current_hit_rate = cache_info['hit_rate']
        
        if current_hit_rate < target_hit_rate and self.cache_size < 5000:
            # Increase cache size
            old_size = self.cache_size
            self.cache_size = min(self.cache_size * 2, 5000)
            logger.info(f"Increased cache size from {old_size} to {self.cache_size} "
                       f"(hit rate: {current_hit_rate:.2%})")
        elif current_hit_rate > 0.95 and self.cache_size > 500:
            # Decrease cache size to save memory
            old_size = self.cache_size
            self.cache_size = max(self.cache_size // 2, 500)
            logger.info(f"Decreased cache size from {old_size} to {self.cache_size} "
                       f"(hit rate: {current_hit_rate:.2%})")

    def validate_dataset_integrity(self, n_samples=100):
        """Comprehensive dataset integrity validation"""
        logger.info("=== DATASET INTEGRITY VALIDATION ===")
        
        issues = []
        sample_count = min(n_samples, len(self.sample_indices))
        
        def validate_sample(idx):
            """Validate a single sample comprehensively"""
            sample_issues = []
            try:
                file_path, sample_idx = self.sample_indices[idx]
                sample = self._load_sample_cached(file_path, sample_idx)
                
                if sample is None:
                    return [f"Sample {idx}: Failed to load"]
                
                # Validate EEG data
                eeg_data = sample.get('input_features')
                if eeg_data is None:
                    sample_issues.append(f"Sample {idx}: Missing EEG data")
                else:
                    try:
                        eeg_array = np.array(eeg_data)
                        
                        # Shape validation
                        if eeg_array.ndim < 2:
                            sample_issues.append(f"Sample {idx}: Invalid EEG dimensions {eeg_array.shape}")
                        
                        # Data quality validation
                        if not np.isfinite(eeg_array).all():
                            sample_issues.append(f"Sample {idx}: Non-finite EEG values")
                        
                        if np.std(eeg_array) < 1e-6:
                            sample_issues.append(f"Sample {idx}: Zero variance EEG")
                            
                    except Exception as e:
                        sample_issues.append(f"Sample {idx}: EEG processing error - {e}")
                
                # Validate text
                text = sample.get('text', '').strip()
                if not text:
                    sample_issues.append(f"Sample {idx}: Empty text")
                elif len(text) < 2:
                    sample_issues.append(f"Sample {idx}: Text too short")
                
                # Test tokenization
                try:
                    tokenization_result = self._safe_tokenize_enhanced(text)
                    if tokenization_result['decoder_input_ids'].max() >= self.vocab_size:
                        sample_issues.append(f"Sample {idx}: Invalid token IDs")
                except Exception as e:
                    sample_issues.append(f"Sample {idx}: Tokenization error - {e}")
                
            except Exception as e:
                sample_issues.append(f"Sample {idx}: General validation error - {e}")
            
            return sample_issues
        
        # Parallel validation
        indices = np.random.choice(len(self.sample_indices), sample_count, replace=False)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(validate_sample, indices))
        
        # Collect all issues
        for sample_issues in results:
            issues.extend(sample_issues)
        
        # Report results
        if issues:
            logger.warning(f"Found {len(issues)} integrity issues:")
            for issue in issues[:20]:  # Limit output
                logger.warning(f"  {issue}")
            if len(issues) > 20:
                logger.warning(f"  ... and {len(issues) - 20} more issues")
        else:
            logger.info("No integrity issues found - dataset appears healthy")
        
        return {
            'total_issues': len(issues),
            'samples_validated': sample_count,
            'issue_rate': len(issues) / sample_count if sample_count > 0 else 0,
            'issues': issues[:100]  # Return first 100 issues
        }