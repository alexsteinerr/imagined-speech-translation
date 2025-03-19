import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from dataset import get_train_test_loaders, MAX_TARGET_LEN
from model import EEG2TextEnsemble, compute_plv
import torchaudio

def load_tokenizer(local_path="tokenizer", hub_id="IDEA-CCNL/Wenzhong-GPT2-110M"):
    import os
    if os.path.isdir(local_path):
        return AutoTokenizer.from_pretrained(local_path)
    else:
        return AutoTokenizer.from_pretrained(hub_id)

def test_all(model, test_loader, tokenizer):
    device = next(model.parameters()).device
    model.eval()
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=1000, n_fft=256, win_length=200, hop_length=100, n_mels=128
    )
    sample_count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            eeg = batch['eeg'].to(device)
            if eeg.dim() == 4:
                eeg = eeg.squeeze(1)
            eeg_lstm = eeg.permute(0,2,1)
            eeg_avg = eeg.mean(dim=1)  # [B, time]
            spec = mel_spec_transform(eeg_avg)
            spec = torch.log(spec + 1e-6)
            spec = spec.unsqueeze(1).repeat(1,3,1,1)
            spec = F.interpolate(spec, size=(224,224), mode='bilinear', align_corners=False)
            final_logits, _, _, _, _ = model(eeg_lstm, spec, eeg)
            preds = torch.argmax(final_logits, dim=-1)
            for i in range(preds.size(0)):
                sample_count += 1
                generated = tokenizer.decode(preds[i].unsqueeze(0), skip_special_tokens=True)
                print(f"Sample {sample_count}:")
                print("Generated:", generated)
                print("Groundtruth:", batch['groundtruth'][i])
                print("-" * 50)

def main():
    dataset_dir = "ds005170-1.1.2/derivatives/preprocessed_pkl"
    batch_size = 16
    _, test_loader, _ = get_train_test_loaders(dataset_dir, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(local_path="tokenizer", hub_id="uer/gpt2-chinese-cluecorpussmall")
    vocab_size = len(tokenizer)
    print("Tokenizer vocab size:", vocab_size)
    model = EEG2TextEnsemble(vocab_size=vocab_size, feature_size=768).to(device)
    checkpoint = torch.load("Ensemble_EEG2TextModel.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    test_all(model, test_loader, tokenizer)

if __name__ == "__main__":
    main()
