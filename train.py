import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_train_test_loaders, MAX_TARGET_LEN
from model import EEG2TextEnsemble, compute_plv
from transformers import AutoTokenizer
from torch.optim import AdamW
import torch.nn.utils as utils
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
epochs = 100
learning_rate = 1e-5 
weight_decay = 1e-4
feature_size = 768

mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=1000,
    n_fft=256,
    win_length=200,
    hop_length=100,
    n_mels=128
)

dataset_dir = "ds005170-1.1.2/derivatives/preprocessed_pkl"
train_loader, test_loader, dataset = get_train_test_loaders(dataset_dir, batch_size=batch_size)

tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Wenzhong-GPT2-110M")
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})
vocab_size = len(tokenizer) 
print("Tokenizer vocab size:", vocab_size)

model = EEG2TextEnsemble(vocab_size=vocab_size, feature_size=feature_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        eeg = batch['eeg'].to(device)
        if eeg.dim() == 4:
            eeg = eeg.squeeze(1)
    
        eeg_lstm = eeg.permute(0, 2, 1)
    
        eeg_avg = eeg.mean(dim=1)
        spec = mel_spec_transform(eeg_avg) 
        spec = torch.log(spec + 1e-6)
        spec = spec.unsqueeze(1).repeat(1, 3, 1, 1)
        spec = nn.functional.interpolate(spec, size=(224,224), mode='bilinear', align_corners=False)
        connectivity_matrix = compute_plv(eeg)
        target_ids = batch['target_ids'].to(device) 

        targets = target_ids[:, 1] if target_ids.size(1) > 1 else target_ids[:, 0]
        optimizer.zero_grad()
        final_logits, _, _, _, _ = model(eeg_lstm, spec, eeg)
        loss = criterion(final_logits, targets)
        loss.backward()
        utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "Ensemble_EEG2TextModel.pth")
tokenizer.save_pretrained("tokenizer")
print("\nTraining complete. Model and tokenizer saved.")