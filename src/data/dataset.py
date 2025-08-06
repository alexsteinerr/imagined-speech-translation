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
    EEG dataset with full preprocessing and regional statistics.
    """
    
    def __init__(self, data_dir, csv_path, tokenizer, max_length=64, eps=1e-6, 
                 max_samples=None, data_augmentation=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eps = eps
        self.max_samples = max_samples
        self.data_augmentation = data_augmentation

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

        # Data loading and full preprocessing
        print("Loading and preprocessing entire dataset...")
        self.data_files = self._get_validated_data_files(data_dir)
        
        # Preprocess and store all data
        self.preprocessed_data = self._preprocess_entire_dataset()
        
        # Compute and print regional statistics
        self.regional_stats = self._compute_regional_stats()
        self._print_regional_stats()

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

    def _preprocess_entire_dataset(self):
        """Preprocess and store the entire dataset."""
        all_samples = []
        all_eeg_data = {region: [] for region in self.region_indices}
        valid_samples = 0
        
        # First pass: collect all valid samples and EEG data for normalization
        logger.info("First pass: collecting all EEG data for normalization...")
        
        for file_path in self.data_files:
            try:
                with open(file_path, 'rb') as f:
                    loaded = pickle.load(f)
                
                if isinstance(loaded, list):
                    samples = loaded
                else:
                    samples = [loaded]
                
                for sample in samples:
                    if self._validate_sample(sample):
                        all_samples.append(sample)
                        
                        # Extract EEG data for each region
                        eeg = np.array(sample['input_features'], dtype=np.float32).squeeze()
                        if eeg.ndim == 1:
                            eeg = eeg.reshape(1, -1)
                        elif eeg.ndim > 2:
                            eeg = eeg.reshape(eeg.shape[0], -1)
                        
                        # Clean extreme values
                        if not np.isfinite(eeg).all():
                            eeg = np.nan_to_num(eeg, nan=0.0, posinf=10.0, neginf=-10.0)
                        
                        # Store regional data
                        for region_name, indices in self.region_indices.items():
                            try:
                                region_eeg = eeg[indices].astype(np.float32)
                                all_eeg_data[region_name].append(region_eeg)
                            except IndexError:
                                logger.warning(f"Index error for region {region_name}")
                                continue
                        
                        valid_samples += 1
                        if self.max_samples and valid_samples >= self.max_samples:
                            break
                    
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
            
            if self.max_samples and valid_samples >= self.max_samples:
                break
        
        logger.info(f"Collected {valid_samples} valid samples for preprocessing")
        
        # Second pass: compute normalization parameters
        logger.info("Computing normalization parameters for each region...")
        self.scalers = {}
        
        for region_name, data_list in all_eeg_data.items():
            if data_list:
                # Concatenate all data for this region
                all_region_data = np.concatenate(data_list, axis=1).T  # (time_samples, channels)
                
                # Fit scaler
                self.scalers[region_name] = RobustScaler(quantile_range=(5.0, 95.0))
                self.scalers[region_name].fit(all_region_data)
                logger.info(f"Fitted scaler for {region_name} region with {all_region_data.shape}")
        
        # Third pass: normalize and augment all samples
        logger.info("Normalizing and augmenting all samples...")
        preprocessed_samples = []
        
        for i, sample in enumerate(all_samples):
            try:
                # Process EEG
                eeg_regions = self._normalize_eeg_sample(sample['input_features'])
                
                # Apply augmentation if enabled
                if self.data_augmentation:
                    eeg_regions = self._augment_eeg_regions(eeg_regions)
                
                # Process text
                text = sample.get('text', '')
                if not text or len(text.strip()) == 0:
                    text = "数据样本"
                
                tokenization = self._safe_tokenize(text)
                
                preprocessed_sample = {
                    'eeg': eeg_regions,
                    **tokenization
                }
                
                preprocessed_samples.append(preprocessed_sample)
                
            except Exception as e:
                logger.error(f"Error preprocessing sample {i}: {e}")
                continue
        
        logger.info(f"Successfully preprocessed {len(preprocessed_samples)} samples")
        return preprocessed_samples

    def _normalize_eeg_sample(self, eeg_data):
        """Normalize a single EEG sample using fitted scalers."""
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
            
        except Exception as e:
            logger.error(f"EEG normalization failed: {e}")
            # Return safe dummy data
            return [np.zeros((len(self.region_indices[region]), 125), dtype=np.float32)
                   for region in ['frontal', 'temporal', 'central', 'parietal']]

    def _augment_eeg_regions(self, eeg_regions):
        """Apply augmentation to EEG regions."""
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

    def _compute_regional_stats(self):
        """Compute comprehensive statistics for each brain region."""
        logger.info("Computing regional statistics...")
        
        regional_stats = {}
        
        for region_idx, region_name in enumerate(['frontal', 'temporal', 'central', 'parietal']):
            region_data_list = []
            
            # Collect all data for this region
            for sample in self.preprocessed_data:
                region_data = sample['eeg'][region_idx]
                region_data_list.append(region_data)
            
            if region_data_list:
                # Stack all samples for this region
                all_region_data = np.stack(region_data_list, axis=0)  # (samples, channels, time)
                
                # Compute statistics
                mean_across_samples = np.mean(all_region_data, axis=0)  # (channels, time)
                std_across_samples = np.std(all_region_data, axis=0)   # (channels, time)
                
                # Overall statistics
                overall_mean = np.mean(all_region_data)
                overall_std = np.std(all_region_data)
                
                # Per-channel statistics
                channel_means = np.mean(all_region_data, axis=(0, 2))  # Mean across samples and time
                channel_stds = np.std(all_region_data, axis=(0, 2))    # Std across samples and time
                
                # Temporal statistics
                temporal_means = np.mean(all_region_data, axis=(0, 1))  # Mean across samples and channels
                temporal_stds = np.std(all_region_data, axis=(0, 1))    # Std across samples and channels
                
                regional_stats[region_name] = {
                    'overall_mean': overall_mean,
                    'overall_std': overall_std,
                    'channel_means': channel_means,
                    'channel_stds': channel_stds,
                    'temporal_means': temporal_means,
                    'temporal_stds': temporal_stds,
                    'shape': all_region_data.shape,
                    'num_channels': len(self.region_indices[region_name]),
                    'channel_names': [self.ch_names[i] for i in self.region_indices[region_name]]
                }
        
        return regional_stats

    def _print_regional_stats(self):
        """Print comprehensive regional statistics."""
        print("\n" + "="*80)
        print("REGIONAL EEG STATISTICS AFTER NORMALIZATION AND AUGMENTATION")
        print("="*80)
        
        for region_name, stats in self.regional_stats.items():
            print(f"\n{region_name.upper()} REGION:")
            print(f"  Channels: {stats['num_channels']} - {stats['channel_names']}")
            print(f"  Data Shape: {stats['shape']} (samples, channels, time)")
            print(f"  Overall Mean: {stats['overall_mean']:.6f}")
            print(f"  Overall Std:  {stats['overall_std']:.6f}")
            
            print(f"  \n  Per-Channel Statistics:")
            for i, (ch_mean, ch_std) in enumerate(zip(stats['channel_means'], stats['channel_stds'])):
                ch_name = stats['channel_names'][i]
                print(f"    {ch_name:>6}: Mean={ch_mean:>8.6f}, Std={ch_std:>8.6f}")
            
            print(f"  \n  Temporal Profile (first 10 timepoints):")
            for i in range(min(10, len(stats['temporal_means']))):
                print(f"    t={i:>2}: Mean={stats['temporal_means'][i]:>8.6f}, Std={stats['temporal_stds'][i]:>8.6f}")
            print(f"    ... (showing first 10 of {len(stats['temporal_means'])} timepoints)")
            
            print(f"  \n  Summary:")
            print(f"    Mean range across channels: {np.min(stats['channel_means']):.6f} to {np.max(stats['channel_means']):.6f}")
            print(f"    Std range across channels:  {np.min(stats['channel_stds']):.6f} to {np.max(stats['channel_stds']):.6f}")
            print(f"    Mean range across time:     {np.min(stats['temporal_means']):.6f} to {np.max(stats['temporal_means']):.6f}")
            print(f"    Std range across time:      {np.min(stats['temporal_stds']):.6f} to {np.max(stats['temporal_stds']):.6f}")
            print("-" * 60)
        
        print(f"\nDATASET SUMMARY:")
        print(f"Total preprocessed samples: {len(self.preprocessed_data)}")
        print(f"Data augmentation: {'Enabled' if self.data_augmentation else 'Disabled'}")
        print(f"Normalization: RobustScaler with quantile_range=(5.0, 95.0)")
        print("="*80 + "\n")

    def _safe_tokenize(self, text):
        """Safe tokenization with comprehensive validation."""
        try:
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
                logger.error(f"Invalid token IDs: max={max_id}, min={min_id}")
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
            logger.error(f"Safe tokenization failed: {e}")
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

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        """Get preprocessed sample."""
        if idx >= len(self.preprocessed_data):
            logger.error(f"Index {idx} out of range")
            # Create fallback sample
            eeg_regions = [np.zeros((len(self.region_indices[region]), 125), dtype=np.float32)
                          for region in ['frontal', 'temporal', 'central', 'parietal']]
            fallback_tokenization = self._create_fallback_tokenization()
            return {'eeg': eeg_regions, **fallback_tokenization}
        
        return self.preprocessed_data[idx]

    def get_sample_stats(self):
        """Get dataset statistics."""
        return {
            'total_samples': len(self.preprocessed_data),
            'preprocessing_mode': 'full_dataset_normalized_and_augmented',
            'normalization': 'RobustScaler(quantile_range=(5.0, 95.0))',
            'augmentation_enabled': self.data_augmentation,
            'regional_stats': self.regional_stats,
            'region_channel_counts': self.region_channel_counts
        }

    def get_regional_stats(self):
        """Get regional statistics."""
        return self.regional_stats