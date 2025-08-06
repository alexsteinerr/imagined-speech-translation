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
    """
    Enhanced EEG dataset with gentler preprocessing to preserve signal information.
    """
    
    def __init__(self, data_dir, csv_path, tokenizer, max_length=64, eps=1e-6, 
                 max_samples=None, data_augmentation=True, text_preprocessing=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eps = eps
        self.max_samples = max_samples
        self.data_augmentation = data_augmentation
        self.text_preprocessing = text_preprocessing

        # Store vocabulary size for validation
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

        # Safe tokenizer setup
        self._setup_tokenizer_safe()

        # Enhanced data loading
        print("Setting up enhanced data loading with gentler preprocessing...")
        self.data_files = self._get_validated_data_files(data_dir)
        self.sample_indices = self._build_robust_sample_index()
        
        # Text preprocessing patterns (more conservative)
        if self.text_preprocessing:
            self._setup_conservative_text_preprocessing()

        # Initialize scalers with gentler parameters
        self.scalers = defaultdict(lambda: RobustScaler(quantile_range=(10.0, 90.0)))  # Gentler quantile range
        
        # Precompute scalers on a subset
        self._precompute_gentle_scalers()

    def _setup_conservative_text_preprocessing(self):
        """Setup very conservative text preprocessing to preserve meaning."""
        self.text_patterns = [
            # Only remove excessive whitespace
            (re.compile(r'\s{2,}'), ' '),
            # Very conservative character filtering - keep more punctuation
            (re.compile(r'[^\u4e00-\u9fff\w\s\.,!?;:\-()""''、。，！？；：（）【】《》]'), ''),
        ]
        logger.info("Setup conservative text preprocessing patterns")

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
        """Safe tokenizer setup without vocabulary modifications."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Set pad_token_id to {self.tokenizer.pad_token_id}")

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
                    pickle.load(f)
                valid_files.append(file_path)
            except Exception as e:
                logger.warning(f"Skipping corrupted file {file_path}: {e}")
        
        logger.info(f"Found {len(valid_files)} valid data files out of {len(files)} total")
        return valid_files

    def _build_robust_sample_index(self):
        """Build sample index with error handling"""
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
                        if self._validate_sample(loaded[i]):
                            sample_indices.append((file_path, i))
                            total_samples += 1
                            if self.max_samples and total_samples >= self.max_samples:
                                break
                else:
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
        """Validate sample structure and content"""
        if not isinstance(sample, dict):
            return False
        
        required_fields = ['input_features', 'text']
        if not all(field in sample for field in required_fields):
            return False
            
        # Validate EEG data dimensions
        eeg_data = sample['input_features']
        if not isinstance(eeg_data, (list, np.ndarray)):
            return False
            
        # Check dimensions
        eeg_array = np.array(eeg_data)
        if len(eeg_array.shape) < 2 or eeg_array.shape[1] != 125:
            logger.warning(f"Invalid EEG shape: {eeg_array.shape}")
            return False
            
        return True

    def _precompute_gentle_scalers(self, subset_size=300):
        """Precompute scalers with gentler normalization."""
        if not self.sample_indices:
            logger.warning("No samples for precomputing scalers")
            return

        region_data = {region: [] for region in self.region_indices}
        sample_count = min(subset_size, len(self.sample_indices))
        indices = np.random.choice(len(self.sample_indices), sample_count, replace=False)
        
        logger.info(f"Precomputing gentle scalers on {sample_count} samples...")
        
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
            
            # Gentle cleaning - only handle extreme cases
            if not np.isfinite(eeg).all():
                eeg = np.nan_to_num(eeg, nan=0.0, posinf=5.0, neginf=-5.0)  # Less aggressive
            
            # Process each region with minimal preprocessing
            for region_name, indices in self.region_indices.items():
                try:
                    region_eeg = eeg[indices].astype(np.float32)
                    region_data[region_name].append(region_eeg)
                except IndexError:
                    logger.warning(f"Index error for region {region_name}")
                    continue
            
            successful_samples += 1
        
        # Fit gentler scalers
        for region_name, data_list in region_data.items():
            if not data_list:
                continue
                
            try:
                # Concatenate along time dimension
                all_data = np.concatenate(data_list, axis=1).T  # (time_samples, channels)
                if all_data.size > 0:
                    self.scalers[region_name].fit(all_data)
                    logger.info(f"Fitted gentle scaler for {region_name} with {all_data.shape[0]} samples")
            except Exception as e:
                logger.warning(f"Failed to fit scaler for {region_name}: {e}")
        
        logger.info(f"Gentle scalers fitted on {successful_samples} samples")

    def _preprocess_text_gently(self, text):
        """Very gentle text preprocessing to preserve meaning."""
        if not self.text_preprocessing or not text:
            return text
        
        try:
            # Apply minimal normalization
            for pattern, replacement in self.text_patterns:
                text = pattern.sub(replacement, text)
            
            # Basic cleanup
            text = text.strip()
            
            # Ensure text is not empty
            if not text:
                text = "文本"  # "text" in Chinese
                
            return text
            
        except Exception as e:
            logger.warning(f"Gentle text preprocessing failed: {e}")
            return text if text else "文本"

    def _safe_tokenize(self, text):
        """Safe tokenization with comprehensive validation."""
        try:
            # Gentle text preprocessing
            text = self._preprocess_text_gently(text)
            
            # Tokenize with validation
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
            min_id = input_ids.min().item()
            
            if max_id >= self.vocab_size or min_id < 0:
                logger.error(f"Invalid token IDs: max={max_id}, min={min_id} for text: '{text}'")
                return self._create_fallback_tokenization()
            
            # Create decoder input IDs
            decoder_start_token_id = (
                self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None
                else self.tokenizer.eos_token_id
            )
            
            if decoder_start_token_id >= self.vocab_size:
                decoder_start_token_id = self.tokenizer.eos_token_id
            
            decoder_input_ids = shift_tokens_right(
                input_ids.unsqueeze(0),
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=decoder_start_token_id
            ).squeeze(0)
            
            # Validate decoder IDs
            max_decoder_id = decoder_input_ids.max().item()
            if max_decoder_id >= self.vocab_size:
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
            logger.error(f"Safe tokenization failed for text '{text}': {e}")
            return self._create_fallback_tokenization()

    def _create_fallback_tokenization(self):
        """Create minimal safe tokenization."""
        try:
            safe_id = min(self.tokenizer.eos_token_id, self.vocab_size - 1)
            input_ids = torch.tensor([safe_id] + [self.tokenizer.pad_token_id] * (self.max_length - 1))
            attention_mask = torch.tensor([1] + [0] * (self.max_length - 1))
            decoder_input_ids = torch.tensor([safe_id] + [self.tokenizer.pad_token_id] * (self.max_length - 1))
            labels = torch.tensor([safe_id] + [-100] * (self.max_length - 1))
            
            return {
                'decoder_input_ids': decoder_input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
            }
        except Exception as e:
            logger.error(f"Fallback tokenization failed: {e}")
            return {
                'decoder_input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.full((self.max_length,), -100, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            }

    def _gentle_eeg_preprocessing(self, eeg_data):
        """Much gentler EEG preprocessing to preserve signal information."""
        try:
            eeg = np.array(eeg_data, dtype=np.float32).squeeze()
            
            # Handle shape issues
            if eeg.ndim == 1:
                eeg = eeg.reshape(1, -1)
            elif eeg.ndim > 2:
                eeg = eeg.reshape(eeg.shape[0], -1)
            
            # Very gentle cleaning - only extreme outliers
            if not np.isfinite(eeg).all():
                logger.debug("Cleaning non-finite values")
                eeg = np.nan_to_num(eeg, nan=0.0, posinf=10.0, neginf=-10.0)  # Less aggressive
            
            # Much gentler outlier handling - preserve more signal variation
            percentile_95 = np.percentile(eeg, 95)
            percentile_5 = np.percentile(eeg, 5)
            iqr = percentile_95 - percentile_5
            
            # Only clip extreme outliers (beyond 4 IQR from median)
            if iqr > 0:
                median_val = np.median(eeg)
                lower_bound = median_val - 4 * iqr
                upper_bound = median_val + 4 * iqr
                eeg = np.clip(eeg, lower_bound, upper_bound)
            
            # Process each region with gentler normalization
            processed_regions = []
            for region_name in ['frontal', 'temporal', 'central', 'parietal']:
                indices = self.region_indices[region_name]
                
                try:
                    region_data = eeg[indices].astype(np.float32)
                except IndexError:
                    logger.warning(f"Index error for region {region_name}")
                    region_data = np.zeros((len(indices), eeg.shape[1]), dtype=np.float32)
                
                # Apply gentle scaling if available
                if region_data.size > 0 and hasattr(self.scalers[region_name], 'scale_'):
                    try:
                        # Use gentler scaling
                        region_data_scaled = self.scalers[region_name].transform(region_data.T).T
                        # Mix scaled and original data to preserve some raw signal
                        region_data = 0.7 * region_data_scaled + 0.3 * self._simple_standardize(region_data)
                    except Exception as e:
                        logger.warning(f"Gentle scaling failed for {region_name}: {e}")
                        region_data = self._simple_standardize(region_data)
                else:
                    region_data = self._simple_standardize(region_data)
                
                # Very light data augmentation
                if self.data_augmentation and np.random.rand() < 0.03:  # Very low probability
                    region_data = self._apply_minimal_augmentation(region_data)
                
                processed_regions.append(region_data)
            
            return processed_regions
            
        except Exception as e:
            logger.error(f"Gentle EEG preprocessing failed: {e}")
            # Return safe dummy data
            return [np.zeros((len(self.region_indices[region]), 1651), dtype=np.float32)
                   for region in ['frontal', 'temporal', 'central', 'parietal']]

    def _simple_standardize(self, data):
        """Simple per-region standardization that preserves signal structure."""
        if data.size == 0:
            return data
        
        # Compute statistics per channel
        mean_vals = np.mean(data, axis=1, keepdims=True)
        std_vals = np.std(data, axis=1, keepdims=True) + 1e-8
        
        # Gentle standardization - don't scale down too much
        standardized = (data - mean_vals) / std_vals
        
        # Scale down less aggressively to preserve signal amplitude relationships
        return standardized * 0.7

    def _apply_minimal_augmentation(self, region_data):
        """Apply very minimal augmentation to preserve signal integrity."""
        try:
            augmented = region_data.copy()
            
            # Extremely light noise (1% of signal std)
            if np.random.rand() < 0.3:
                signal_std = max(np.std(augmented) * 0.01, 1e-7)  # Even smaller
                noise = np.random.normal(0, signal_std, augmented.shape)
                augmented += noise
            
            # Very small amplitude scaling (±2%)
            if np.random.rand() < 0.2:
                scale_factor = np.random.uniform(0.98, 1.02)  # Smaller range
                augmented *= scale_factor
            
            return augmented
        except Exception as e:
            logger.warning(f"Minimal augmentation failed: {e}")
            return region_data

    def load_single_sample(self, file_path, sample_idx):
        """Load single sample with error handling."""
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
        """Create fallback sample with proper dimensions."""
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
        """Enhanced getitem with gentle preprocessing and comprehensive error handling."""
        if idx >= len(self.sample_indices):
            logger.error(f"Index {idx} out of range")
            return self._create_fallback_sample()

        file_path, sample_idx = self.sample_indices[idx]
        sample = self.load_single_sample(file_path, sample_idx)

        if sample is None:
            logger.warning(f"Failed to load sample {idx}, using fallback")
            return self._create_fallback_sample()

        try:
            # Process EEG data with gentle preprocessing
            eeg_regions = self._gentle_eeg_preprocessing(sample['input_features'])
            
            # Process text gently
            text = sample.get('text', '')
            if not text or len(text.strip()) == 0:
                text = "数据样本"  # "data sample" in Chinese
            
            # Safe tokenization
            tokenization_result = self._safe_tokenize(text)
            
            return {
                'eeg': eeg_regions,
                **tokenization_result
            }
            
        except Exception as e:
            logger.error(f"Sample processing failed for index {idx}: {e}")
            return self._create_fallback_sample()

    def get_sample_stats(self):
        """Get enhanced dataset statistics."""
        if not hasattr(self, '_stats'):
            logger.info("Computing enhanced dataset statistics...")
            
            text_lengths = []
            eeg_shapes = []
            valid_samples = 0
            unique_texts = set()
            
            # Sample subset for statistics
            sample_size = min(1000, len(self.sample_indices))
            indices = np.random.choice(len(self.sample_indices), sample_size, replace=False)
            
            for idx in indices:
                try:
                    file_path, sample_idx = self.sample_indices[idx]
                    sample = self.load_single_sample(file_path, sample_idx)
                    
                    if sample and self._validate_sample(sample):
                        text = sample.get('text', '')
                        if text:
                            # Tokenized length
                            try:
                                tokens = self.tokenizer.tokenize(text)
                                text_lengths.append(len(tokens))
                                unique_texts.add(text.strip())
                            except:
                                text_lengths.append(len(text.split()))
                        
                        eeg_data = sample.get('input_features')
                        if eeg_data is not None:
                            eeg_shapes.append(np.array(eeg_data).shape)
                        
                        valid_samples += 1
                        
                except Exception:
                    continue
            
            # Compute text diversity
            text_diversity = len(unique_texts) / len(text_lengths) if text_lengths else 0.0
            
            self._stats = {
                'total_samples': len(self.sample_indices),
                'valid_samples': valid_samples,
                'avg_text_length': np.mean(text_lengths) if text_lengths else 0,
                'text_length_std': np.std(text_lengths) if text_lengths else 0,
                'text_diversity': text_diversity,
                'unique_texts': len(unique_texts),
                'unique_eeg_shapes': len(set(tuple(shape) for shape in eeg_shapes)),
                'most_common_eeg_shape': max(set(tuple(shape) for shape in eeg_shapes), 
                                           key=lambda x: [tuple(shape) for shape in eeg_shapes].count(x)) if eeg_shapes else None,
                'region_channel_counts': self.region_channel_counts,
                'preprocessing_mode': 'gentle'
            }
            
            logger.info(f"Enhanced dataset stats: {self._stats}")
        
        return self._stats

    def get_preprocessing_info(self):
        """Get information about preprocessing settings."""
        return {
            'data_augmentation': self.data_augmentation,
            'text_preprocessing': self.text_preprocessing,
            'augmentation_probability': 0.03,
            'scaling_method': 'gentle_robust',
            'outlier_handling': 'percentile_based',
            'signal_preservation': 'high'
        }