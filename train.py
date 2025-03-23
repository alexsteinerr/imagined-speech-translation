import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm

MAX_TARGET_LEN = 32

class EEGDataset(Dataset):
    def __init__(self, directory, max_seq_len=2500, tokenizer_name="uer/gpt2-chinese-cluecorpussmall"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
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
        for fp in tqdm(all_files, desc="Loading EEG dataset"):
            data = self.load_single_eeg_file(fp)
            if data is None:
                continue
            if isinstance(data, list):
                data = data[0]
            eeg = torch.tensor(data['input_features'], dtype=torch.float32)
            eeg = self.pad_or_truncate(eeg)
            label = data['text']
            tokenized = self.tokenizer(label, padding='max_length', truncation=True, max_length=MAX_TARGET_LEN - 1, return_tensors='pt')
            input_ids = tokenized['input_ids'].squeeze(0)
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            input_ids = torch.cat([torch.tensor([bos], dtype=torch.long), input_ids], dim=0)
            if input_ids.size(0) < MAX_TARGET_LEN:
                pad_len = MAX_TARGET_LEN - input_ids.size(0)
                pad = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                input_ids = torch.cat([input_ids, pad], dim=0)
            elif input_ids.size(0) > MAX_TARGET_LEN:
                input_ids = input_ids[:MAX_TARGET_LEN]
            self.samples.append({'eeg': eeg, 'target_ids': input_ids, 'groundtruth': label})
        random.shuffle(self.samples)
        self.build_label_mapping()

    def load_single_eeg_file(self, fp):
        try:
            with open(fp, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            return None

    def pad_or_truncate(self, eeg):
        L = eeg.shape[-1]
        if L < self.max_seq_len:
            pad = torch.zeros((eeg.shape[0], eeg.shape[1], self.max_seq_len - L))
            return torch.cat((eeg, pad), dim=-1)
        elif L > self.max_seq_len:
            return eeg[..., :self.max_seq_len]
        return eeg

    def build_label_mapping(self):
        self.label_to_idx = {}
        idx = 0
        for s in self.samples:
            label = s['groundtruth']
            if label and label not in self.label_to_idx:
                self.label_to_idx[label] = idx
                idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_train_test_loaders(directory, batch_size=16, test_split=0.2, max_seq_len=2500):
    dataset = EEGDataset(directory, max_seq_len=max_seq_len)
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, dataset

class ViTBranch(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, num_layers, num_heads, vocab_size):
        super(ViTBranch, self).__init__()
        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size, embed_dim, _, _ = x.size()
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        cls_out = x[:, 0, :]
        logits = self.fc(cls_out)
        return F.log_softmax(logits, dim=-1)

class LSTMBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, vocab_size, bidirectional=True):
        super(LSTMBranch, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, vocab_size)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        x = x.view(batch_size, C * H, W).permute(0, 2, 1)
        output, (hn, _) = self.lstm(x)
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=-1)
        else:
            last_hidden = hn[-1]
        logits = self.fc(last_hidden)
        return F.log_softmax(logits, dim=-1)

class GNNBranch(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_layers, vocab_size):
        super(GNNBranch, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        x_nodes = x.view(batch_size, C, -1).mean(dim=-1)
        x_nodes = x_nodes.unsqueeze(-1)
        adj = torch.ones(batch_size, C, C, device=x.device) / C
        for layer in self.layers:
            x_nodes = F.relu(torch.bmm(adj, layer(x_nodes)))
        graph_embedding = x_nodes.mean(dim=1)
        logits = self.fc(graph_embedding.squeeze(-1))
        return F.log_softmax(logits, dim=-1)

def ensemble_predictions(preds, losses):
    weights = [1.0 / loss for loss in losses]
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]
    probs = [p.exp() for p in preds]
    ensemble_prob = sum(w * p for w, p in zip(norm_weights, probs))
    return torch.log(ensemble_prob + 1e-8)

def train_branch(model, dataloader, optimizer, criterion, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            eeg = batch['eeg'].to(device)
            target_ids = batch['target_ids'].to(device)
            optimizer.zero_grad()
            output = model(eeg)
            loss = criterion(output, target_ids[:, 0])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return avg_loss

def predict_ensemble(eeg, models, losses, device='cuda'):
    preds = []
    for model in models:
        model.to(device)
        model.eval()
        with torch.no_grad():
            preds.append(model(eeg.to(device)))
    ensemble_log_prob = ensemble_predictions(preds, losses)
    return ensemble_log_prob

def main():
    directory = "/content/eeg"
    batch_size = 8
    test_split = 0.2
    max_seq_len = 2500
    in_channels = 1
    embed_dim = 128
    patch_size = 4
    num_layers_vit = 6
    num_heads = 8
    vocab_size = 5000
    hidden_dim_lstm = 256
    num_layers_lstm = 3
    node_feature_dim = 1
    hidden_dim_gnn = 128
    num_layers_gnn = 3

    train_loader, test_loader, dataset = get_train_test_loaders(directory, batch_size, test_split, max_seq_len)
    model_vit = ViTBranch(in_channels, embed_dim, patch_size, num_layers_vit, num_heads, vocab_size)
    sample_eeg = dataset[0]['eeg']
    C, H, W = sample_eeg.shape
    input_dim_lstm = C * H
    model_lstm = LSTMBranch(input_dim_lstm, hidden_dim_lstm, num_layers_lstm, vocab_size)
    model_gnn = GNNBranch(node_feature_dim, hidden_dim_gnn, num_layers_gnn, vocab_size)
    criterion = nn.NLLLoss()
    optimizer_vit = optim.Adam(model_vit.parameters(), lr=1e-4)
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=1e-4)
    optimizer_gnn = optim.Adam(model_gnn.parameters(), lr=1e-4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm.write("Training ViT branch...")
    loss_vit = train_branch(model_vit, train_loader, optimizer_vit, criterion, num_epochs=20, device=device)
    tqdm.write("Training LSTM branch...")
    loss_lstm = train_branch(model_lstm, train_loader, optimizer_lstm, criterion, num_epochs=20, device=device)
    tqdm.write("Training GNN branch...")
    loss_gnn = train_branch(model_gnn, train_loader, optimizer_gnn, criterion, num_epochs=20, device=device)
    losses = [loss_vit, loss_lstm, loss_gnn]
    models_list = [model_vit, model_lstm, model_gnn]
    for batch in tqdm(test_loader, desc="Evaluating ensemble"):
        eeg_batch = batch['eeg']
        ensemble_log_prob = predict_ensemble(eeg_batch, models_list, losses, device=device)
        print("Ensemble prediction (log-probs) shape:", ensemble_log_prob.shape)
        break

if __name__ == "__main__":
    main()