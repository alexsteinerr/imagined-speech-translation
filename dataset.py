import os
import pickle
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm

MAX_TARGET_LEN = 32

class EEGDataset(Dataset):
    def __init__(self, directory, max_seq_len=2500):
        self.tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Wenzhong-GPT2-110M")
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '[EOS]'})
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': '[BOS]'})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_len = max_seq_len

        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".pkl"):
                    all_files.append(os.path.join(root, file))
        random.shuffle(all_files)

        self.samples = []
        for file_path in tqdm(all_files, desc="Loading EEG dataset"):
            data = self.load_single_eeg_file(file_path)
            if data is None:
                continue
            if isinstance(data, list):
                data = data[0]
            eeg_signal = torch.tensor(data['input_features'], dtype=torch.float32)
            eeg_signal = self.pad_or_truncate(eeg_signal)
            label_text = data['text']
            tokenized = self.tokenizer(label_text, padding='max_length', truncation=True,
                                        max_length=MAX_TARGET_LEN - 1, return_tensors='pt')
            input_ids = tokenized['input_ids'].squeeze(0)
            bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            input_ids = torch.cat([torch.tensor([bos_id], dtype=torch.long), input_ids], dim=0)
            if input_ids.size(0) < MAX_TARGET_LEN:
                pad_length = MAX_TARGET_LEN - input_ids.size(0)
                padding = torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                input_ids = torch.cat([input_ids, padding], dim=0)
            elif input_ids.size(0) > MAX_TARGET_LEN:
                input_ids = input_ids[:MAX_TARGET_LEN]
            self.samples.append({'eeg': eeg_signal, 'target_ids': input_ids, 'groundtruth': label_text})
        random.shuffle(self.samples)
        self.build_label_mapping()

    def load_single_eeg_file(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            return None

    def pad_or_truncate(self, eeg_signal):
        seq_len = eeg_signal.shape[-1]
        if seq_len < self.max_seq_len:
            padding = torch.zeros((eeg_signal.shape[0], eeg_signal.shape[1], self.max_seq_len - seq_len))
            return torch.cat((eeg_signal, padding), dim=-1)
        elif seq_len > self.max_seq_len:
            return eeg_signal[..., :self.max_seq_len]
        return eeg_signal

    def build_label_mapping(self):
        self.label_to_idx = {}
        idx = 0
        for sample in self.samples:
            label_text = sample['groundtruth']
            if label_text and label_text not in self.label_to_idx:
                self.label_to_idx[label_text] = idx
                idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_train_test_loaders(directory, batch_size=16, test_split=0.2):
    dataset = EEGDataset(directory)
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, dataset