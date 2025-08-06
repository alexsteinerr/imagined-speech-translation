import torch
from transformers import AutoTokenizer
from dataset import EEGDataset
from models import EEGDecodingModel
from torch.utils.data import DataLoader, random_split
from sacrebleu.metrics import BLEU
import pandas as pd
from tqdm.auto import tqdm

def run_ablation_study(
    data_dir='eeg_data/',
    montage_file='montage.csv',
    pretrained='fnlp/bart-base-chinese',
    max_length=32,
    hidden_dim=128,
    batch_size=8,
    model_weights='eeg_model.pth'
):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and validation dataset
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    dataset   = EEGDataset(data_dir, montage_file, tokenizer, max_length=max_length)
    n         = len(dataset)
    val_n     = int(0.3 * n)
    _, val_ds = random_split(dataset, [n - val_n, val_n])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Scorers
    bleu1_scorer = BLEU(smooth_method="exp", smooth_value=0.1,
                        effective_order=True, max_ngram_order=1)
    bleu4_scorer = BLEU(smooth_method="exp", smooth_value=0.1,
                        effective_order=True, max_ngram_order=4)

    def evaluate_variant(model, name):
        model.to(device).eval()
        refs, hyps = [], []
        loop = tqdm(val_loader, desc=f"Eval {name}", leave=False)
        with torch.no_grad():
            for batch in loop:
                eeg    = [r.float().to(device) for r in batch['eeg']]
                dec_in = batch['decoder_input_ids'].squeeze(1).to(device)
                feats   = model.brain_encoder(eeg)
                pred_ids = model.bart_decoder.generate_from_eeg(
                    feats, max_length=max_length, num_beams=8
                )
                gt_texts   = tokenizer.batch_decode(dec_in,    skip_special_tokens=True)
                pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                refs.extend(gt_texts)
                hyps.extend(pred_texts)
        b1 = bleu1_scorer.corpus_score(hyps, [refs]).score
        b4 = bleu4_scorer.corpus_score(hyps, [refs]).score
        return {'variant': name,
                'BLEU-1': b1, 'BLEU-4': b4}

    # Load and evaluate full + ablations
    variants = []
    # Full model
    full = EEGDecodingModel(n_timepoints=1651, hidden_dim=hidden_dim)
    full.load_state_dict(torch.load(model_weights, map_location=device), strict=True)
    variants.append(('full', full))

    flags = [
        ('no_cross_region_attn',   {'disable_cross_region_attn':True}),
        ('uniform_region_weight',  {'uniform_region_weight':True}),
        ('no_positional_encoding', {'disable_positional_encoding':True}),
        ('cnn_only',               {'cnn_only':True}),
        ('no_cross_modal',         {'disable_cross_modal':True}),
        ('baseline_cnn',           {'baseline_cnn':True}),
    ]
    for name, kw in flags:
        m = EEGDecodingModel(
            n_timepoints=1651, hidden_dim=hidden_dim, **kw
        )
        m.load_state_dict(torch.load(model_weights, map_location=device), strict=False)
        variants.append((name, m))

    results = []
    for name, m in variants:
        res = evaluate_variant(m, name)
        results.append(res)

    df = pd.DataFrame(results)
    df.to_csv('ablation_results.csv', index=False)
    print(df)

if __name__ == '__main__':
    run_ablation_study()
