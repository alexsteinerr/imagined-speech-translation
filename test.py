import torch
from transformers import AutoTokenizer, BartConfig
from dataset import EEGDataset
from models import EEGDecodingModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ─── Configuration ────────────────────────────────────────────────────────────
data_dir        = 'eeg_data/'
montage_file    = 'montage.csv'
pretrained      = 'fnlp/bart-base-chinese'
max_length      = 32
n_timepoints    = 1651
hidden_dim      = 256              # must match training config
checkpoint_path = 'best_eeg_model.pth'

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─── Tokenizer & Dataset ──────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(pretrained)
dataset   = EEGDataset(data_dir, montage_file, tokenizer, max_length=max_length)
print(f"Loaded {len(dataset)} test samples.")

# ─── Bart Config (must match training) ────────────────────────────────────────
config = BartConfig.from_pretrained(
    pretrained,
    encoder_layers=12,
    decoder_layers=12,
    d_model=1024,
    encoder_ffn_dim=4096,
    decoder_ffn_dim=4096,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    dropout=0.2,
    attention_dropout=0.2,
    encoder_layerdrop=0.1,
    decoder_layerdrop=0.1,
    label_smoothing=0.1,
)

# ─── Model Definition ─────────────────────────────────────────────────────────
model = EEGDecodingModel(
    bart_config=config,
    n_timepoints=n_timepoints,
    hidden_dim=hidden_dim,
    disable_cross_region_attn=False,
    uniform_region_weight=False,
    disable_positional_encoding=False,
    cnn_only=False,
    disable_cross_modal=False
)
model.to(device)

# ─── Load Trained Weights ─────────────────────────────────────────────────────
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print(f"✅ Loaded model from: {checkpoint_path}")

# ─── BLEU Evaluation Setup ────────────────────────────────────────────────────
smooth = SmoothingFunction().method1
refs = []
preds = []

# ─── Inference Over Full Dataset ──────────────────────────────────────────────
for i in range(len(dataset)):
    sample = dataset[i]
    eeg = [torch.tensor(r).unsqueeze(0).float().to(device) for r in sample['eeg']]
    decoder_input_ids = sample['decoder_input_ids'].unsqueeze(0).to(device)

    with torch.no_grad():
        feats = model.brain_encoder(eeg)
        generated_ids = model.bart_decoder.generate_from_eeg(
            feats,
            max_length=max_length,
            num_beams=8
        )

    pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    gt_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

    refs.append([list(gt_text)])   # reference: list of list of characters
    preds.append(list(pred_text))  # prediction: list of characters

    print(f"\n--- Sample {i+1}/{len(dataset)} ---")
    print(f"🎯 Ground Truth          : {gt_text}")
    print(f"🧠 Model Prediction     : {pred_text}")

# ─── Compute BLEU-1 Score ─────────────────────────────────────────────────────
bleu1_scores = [
    sentence_bleu(r, p, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    for r, p in zip(refs, preds)
]
avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)

print(f"\n✅ Average BLEU-1 Score over {len(preds)} samples: {avg_bleu1:.2%}")
