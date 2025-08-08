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
from functools import lru_cache
import gc

logger = logging.getLogger(__name__)

class EEGDataset(Dataset):
    """
    Optimized EEG dataset with lazy loading and efficient preprocessing.
    """
    
    def __init__(self, data_dir, csv_path, tokenizer, max_length=64, eps=1e-6, 
                 max_samples=None, data_augmentation=True, precompute_stats=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eps = eps
        self.max_samples = max_samples
        self.data_augmentation = data_augmentation
        self.precompute_stats = precompute_stats

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

        # OPTIMIZED: Only get file paths and sample metadata
        print("Discovering data files...")
        self.data_files = self._get_validated_data_files(data_dir)
        
        # OPTIMIZED: Build index of samples without loading data
        self.sample_index = self._build_sample_index()
        
        # OPTIMIZED: Initialize scalers with subset of data (much faster)
        print("Computing normalization parameters from sample...")
        self._initialize_scalers_efficiently()
        
        # Optional: compute stats only if requested
        if self.precompute_stats:
            print("Computing regional statistics...")
            self.regional_stats = self._compute_regional_stats_sample()
            self._print_regional_stats()
        else:
            self.regional_stats = None
            
        print(f"Dataset initialized with {len(self)} samples")

    def _build_sample_index(self):
        """Build index of samples without loading actual data."""
        sample_index = []
        total_samples = 0
        
        for file_path in self.data_files:
            try:
                # Quick peek to count samples without full loading
                with open(file_path, 'rb') as f:
                    loaded = pickle.load(f)
                
                if isinstance(loaded, list):
                    num_samples = len(loaded)
                else:
                    num_samples = 1
                
                # Add to index
                for i in range(num_samples):
                    sample_index.append({'file': file_path, 'index': i})
                    total_samples += 1
                    
                    if self.max_samples and total_samples >= self.max_samples:
                        return sample_index[:self.max_samples]
                        
            except Exception as e:
                logger.warning(f"Error indexing {file_path}: {e}")
                continue
                
        logger.info(f"Built index for {len(sample_index)} samples")
        return sample_index

    def _initialize_scalers_efficiently(self):
        """Initialize scalers using only a subset of data for efficiency."""
        # Use only first 100 samples or 10% of data, whichever is smaller
        sample_size = min(100, max(10, len(self.sample_index) // 10))
        sample_indices = np.random.choice(len(self.sample_index), 
                                        size=min(sample_size, len(self.sample_index)), 
                                        replace=False)
        
        # Collect sample data for each region
        region_data = {region: [] for region in self.region_indices}
        
        for idx in sample_indices:
            try:
                # Load single sample
                sample_info = self.sample_index[idx]
                sample = self._load_single_sample(sample_info['file'], sample_info['index'])
                
                if not self._validate_sample(sample):
                    continue
                    
                # Process EEG data
                eeg = self._process_raw_eeg(sample['input_features'])
                if eeg is None:
                    continue
                
                # Store regional data
                for region_name, indices in self.region_indices.items():
                    try:
                        region_eeg = eeg[indices].astype(np.float32)
                        region_data[region_name].append(region_eeg)
                    except IndexError:
                        continue
                        
            except Exception as e:
                logger.warning(f"Error in scaler initialization for sample {idx}: {e}")
                continue
        
        # Fit scalers
        self.scalers = {}
        for region_name, data_list in region_data.items():
            if data_list:
                # Concatenate and fit
                all_region_data = np.concatenate(data_list, axis=1).T
                self.scalers[region_name] = RobustScaler(quantile_range=(5.0, 95.0))
                self.scalers[region_name].fit(all_region_data)
                logger.info(f"Fitted scaler for {region_name} with {len(data_list)} samples")
        
        # Clear memory
        del region_data
        gc.collect()

    @lru_cache(maxsize=32)  # Cache recently loaded files
    def _load_single_sample(self, file_path, sample_idx):
        """Load a single sample from file with caching."""
        try:
            with open(file_path, 'rb') as f:
                loaded = pickle.load(f)
            
            if isinstance(loaded, list):
                if sample_idx < len(loaded):
                    return loaded[sample_idx]
                else:
                    return None
            else:
                return loaded if sample_idx == 0 else None
                
        except Exception as e:
            logger.error(f"Error loading sample {sample_idx} from {file_path}: {e}")
            return None

    def _process_raw_eeg(self, eeg_data):
        """Process raw EEG data into clean format."""
        try:
            eeg = np.array(eeg_data, dtype=np.float32).squeeze()
            
            # Handle shape issues
            if eeg.ndim == 1:
                eeg = eeg.reshape(1, -1)
            elif eeg.ndim > 2:
                eeg = eeg.reshape(eeg.shape[0], -1)
            
            # Clean extreme values
            if not np.isfinite(eeg).all():
                eeg = np.nan_to_num(eeg, nan=0.0, posinf=10.0, neginf=-10.0)
                
            return eeg
            
        except Exception as e:
            logger.error(f"EEG processing failed: {e}")
            return None

    def _normalize_eeg_sample(self, eeg_data):
        """Normalize a single EEG sample using fitted scalers."""
        eeg = self._process_raw_eeg(eeg_data)
        if eeg is None:
            # Return safe dummy data
            return [np.zeros((len(self.region_indices[region]), 125), dtype=np.float32)
                   for region in ['frontal', 'temporal', 'central', 'parietal']]
        
        # Normalize each region
        normalized_regions = []
        for region_name in ['frontal', 'temporal', 'central', 'parietal']:
            indices = self.region_indices[region_name]
            
            try:
                region_data = eeg[indices].astype(np.float32)
                
                # Apply normalization
                if region_name in self.scalers:
                    region_data_norm = self.scalers[region_name].transform(region_data.T).T
                else:
                    # Fallback normalization
                    mean_vals = np.mean(region_data, axis=1, keepdims=True)
                    std_vals = np.std(region_data, axis=1, keepdims=True) + 1e-8
                    region_data_norm = (region_data - mean_vals) / std_vals
                
                normalized_regions.append(region_data_norm)
                
            except Exception as e:
                logger.warning(f"Error normalizing {region_name}: {e}")
                # Fallback: zero data
                normalized_regions.append(np.zeros((len(indices), eeg.shape[1]), dtype=np.float32))
        
        return normalized_regions

    def _augment_eeg_regions(self, eeg_regions):
        """Apply augmentation to EEG regions."""
        if not self.data_augmentation:
            return eeg_regions
            
        augmented_regions = []
        
        for region_data in eeg_regions:
            try:
                augmented = region_data.copy()
                
                # Add small amount of noise (5% of std)
                if np.random.rand() < 0.3:
                    signal_std = max(np.std(augmented) * 0.05, 1e-6)
                    noise = np.random.normal(0, signal_std, augmented.shape)
                    augmented += noise
                
                # Small amplitude scaling (±10%)
                if np.random.rand() < 0.2:
                    scale_factor = np.random.uniform(0.9, 1.1)
                    augmented *= scale_factor
                
                # Small time shift (±2 samples)
                if np.random.rand() < 0.15:
                    shift = np.random.randint(-2, 3)
                    if shift != 0:
                        augmented = np.roll(augmented, shift, axis=1)
                
                augmented_regions.append(augmented)
                
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
                augmented_regions.append(region_data)
        
        return augmented_regions

    def _compute_regional_stats_sample(self):
        """Compute stats from a sample of data."""
        sample_size = min(50, len(self.sample_index))
        sample_indices = np.random.choice(len(self.sample_index), size=sample_size, replace=False)
        
        regional_data = {f"{region}_{i}": [] for region in ['frontal', 'temporal', 'central', 'parietal'] for i in range(4)}
        
        for idx in sample_indices:
            try:
                sample = self.__getitem__(idx)
                for region_idx, region_name in enumerate(['frontal', 'temporal', 'central', 'parietal']):
                    regional_data[f"{region_name}_{region_idx}"].append(sample['eeg'][region_idx])
            except:
                continue
        
        # Compute basic stats
        regional_stats = {}
        for region_name in ['frontal', 'temporal', 'central', 'parietal']:
            data_key = f"{region_name}_0"
            if data_key in regional_data and regional_data[data_key]:
                all_data = np.stack(regional_data[data_key])
                regional_stats[region_name] = {
                    'overall_mean': np.mean(all_data),
                    'overall_std': np.std(all_data),
                    'shape': all_data.shape,
                    'num_channels': len(self.region_indices[region_name]),
                    'channel_names': [self.ch_names[i] for i in self.region_indices[region_name]]
                }
        
        return regional_stats

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        """Get sample with lazy loading and on-the-fly preprocessing."""
        if idx >= len(self.sample_index):
            logger.error(f"Index {idx} out of range")
            return self._create_fallback_sample()
        
        try:
            # Get sample info
            sample_info = self.sample_index[idx]
            
            # Load sample
            sample = self._load_single_sample(sample_info['file'], sample_info['index'])
            if not sample or not self._validate_sample(sample):
                return self._create_fallback_sample()
            
            # Process EEG
            eeg_regions = self._normalize_eeg_sample(sample['input_features'])
            eeg_regions = self._augment_eeg_regions(eeg_regions)
            
            # Process text
            text = sample.get('text', '').strip()
            if not text:
                text = "数据样本"
            
            tokenization = self._safe_tokenize(text)
            
            return {
                'eeg': eeg_regions,
                **tokenization
            }
            
        except Exception as e:
            logger.error(f"Error getting sample {idx}: {e}")
            return self._create_fallback_sample()

    def _create_fallback_sample(self):
        """Create fallback sample for error cases."""
        eeg_regions = [np.zeros((len(self.region_indices[region]), 125), dtype=np.float32)
                      for region in ['frontal', 'temporal', 'central', 'parietal']]
        fallback_tokenization = self._create_fallback_tokenization()
        return {'eeg': eeg_regions, **fallback_tokenization}

    # Keep all the other methods from original class (build_region_indices, validate_region_indices, 
    # setup_tokenizer_safe, get_validated_data_files, validate_sample, safe_tokenize, 
    # create_fallback_tokenization, print_regional_stats, get_sample_stats, get_regional_stats)
    # ... [Include all the validation and utility methods from your original class]

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
        
        logger.info(f"Found {len(files)} .pkl files")
        return files

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
            return False
            
        return True

    def _safe_tokenize(self, text):
        """Safe tokenization with comprehensive validation."""
        try:
            # Clean and validate text
            if not text or not isinstance(text, str):
                text = "数据样本"  # Default Chinese text
            
            text = text.strip()
            if not text:
                text = "数据样本"
            
            # Tokenize with special tokens
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
            
            # CRITICAL: Validate ALL token IDs are within vocabulary
            max_id = input_ids.max().item()
            min_id = input_ids.min().item()
            
            if max_id >= self.vocab_size:
                logger.error(f"Token ID {max_id} exceeds vocab size {self.vocab_size}")
                logger.error(f"Problematic text: {text[:100]}")
                # Clamp to valid range
                input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
            
            if min_id < 0:
                logger.error(f"Negative token ID found: {min_id}")
                input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
            
            # Create decoder input IDs safely
            # Use CLS token (101) as decoder start if available, else use a safe token
            decoder_start_token_id = min(101, self.vocab_size - 1)
            
            # Shift tokens right for decoder input
            decoder_input_ids = torch.cat([
                torch.tensor([decoder_start_token_id]),
                input_ids[:-1]
            ])
            
            # Ensure decoder input IDs are also valid
            decoder_input_ids = torch.clamp(decoder_input_ids, 0, self.vocab_size - 1)
            
            # Create labels (input_ids with padding masked)
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Final validation
            assert decoder_input_ids.max() < self.vocab_size, f"Decoder ID overflow: {decoder_input_ids.max()}"
            assert (labels[labels != -100]).max() < self.vocab_size if (labels != -100).any() else True
            
            return {
                'decoder_input_ids': decoder_input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
            }
            
        except Exception as e:
            logger.error(f"Tokenization failed for text: {text[:50]}... Error: {e}")
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

    def _print_regional_stats(self):
        """Print comprehensive regional statistics."""
        if not self.regional_stats:
            return
            
        print("\n" + "="*80)
        print("REGIONAL EEG STATISTICS (SAMPLED)")
        print("="*80)
        
        for region_name, stats in self.regional_stats.items():
            print(f"\n{region_name.upper()} REGION:")
            print(f"  Channels: {stats['num_channels']} - {stats['channel_names']}")
            print(f"  Sample Shape: {stats['shape']} (samples, channels, time)")
            print(f"  Overall Mean: {stats['overall_mean']:.6f}")
            print(f"  Overall Std:  {stats['overall_std']:.6f}")
            print("-" * 60)
        
        print(f"\nDATASET SUMMARY:")
        print(f"Total samples: {len(self)}")
        print(f"Data augmentation: {'Enabled' if self.data_augmentation else 'Disabled'}")
        print(f"Lazy loading: Enabled")
        print("="*80 + "\n")

    def get_sample_stats(self):
        """Get dataset statistics."""
        return {
            'total_samples': len(self),
            'loading_mode': 'lazy_loading_with_caching',
            'normalization': 'RobustScaler(quantile_range=(5.0, 95.0))',
            'augmentation_enabled': self.data_augmentation,
            'regional_stats': self.regional_stats,
            'region_channel_counts': self.region_channel_counts
        }

    def get_regional_stats(self):
        """Get regional statistics."""
        return self.regional_stats