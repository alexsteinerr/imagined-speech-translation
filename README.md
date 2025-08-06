# Inner Speech Translation

A deep learning model for decoding Chinese text from EEG brain signals using multi-region neural encoding and BART-based text generation.

## Overview

This repository implements a neural decoder that translates EEG brain activity into Chinese text. The model uses:

- **Multi-region EEG encoding**: Processes signals from frontal, temporal, central, and parietal brain regions
- **CNN backbone with attention**: Extracts spatiotemporal features from EEG signals
- **Cross-region fusion**: Combines information across brain regions using transformer attention
- **BART decoder**: Generates Chinese text conditioned on EEG features

## Architecture

```
EEG Signals (4 regions) → Regional CNNs → Cross-region Attention → EEG-BART Projection → Chinese Text
```

### Key Components

1. **BrainRegionEncoder**: Processes EEG from different brain regions independently then fuses them
2. **Conv1DWithAttention**: CNN backbone with optional self-attention for temporal modeling
3. **BARTDecoder**: Chinese BART model conditioned on EEG features for text generation

## Installation

```bash
# Clone repository
git clone https://github.com/alexsteinerr/imagined-speech-translation
cd eeg-to-text

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Data Format

EEG data should be stored as pickle files with the following structure:

```python
{
    'input_features': np.array,  # Shape: (n_channels, n_timepoints)
    'text': str                  # Target Chinese text
}
```

Electrode montage should be provided as CSV:
```csv
label
Fp1
Fp2
F3
F4
...
```

## Training

### Quick Start

```bash
python scripts/train.py
```

### Configuration

Modify `config/training_config.py` to adjust:

- **Model architecture**: Hidden dimensions, attention heads, etc.
- **Training parameters**: Learning rates, batch size, epochs
- **Data paths**: Input data directory and montage file
- **Generation settings**: Beam search, sampling parameters

### Key Configuration Options

```python
CONFIG = {
    'hidden_dim': 768,              # Model hidden dimension
    'batch_size': 4,                # Training batch size
    'accumulation_steps': 8,        # Gradient accumulation
    'brain_encoder_lr': 1e-4,       # EEG encoder learning rate
    'bart_decoder_lr': 1e-5,        # BART decoder learning rate
    'disable_cross_region_attn': False,  # Enable/disable region fusion
    'cnn_only': False,              # Use CNN-only (no attention)
}
```

## Model Variants

The codebase supports several architectural variants:

1. **Full Model**: Multi-region CNN + cross-attention + BART
2. **CNN-Only**: Remove attention layers, use CNN pooling
3. **No Cross-Region**: Process regions independently
4. **Uniform Weighting**: Equal importance for all brain regions

## Evaluation

```python
from src.evaluation.evaluator import ChineseEvaluator

evaluator = ChineseEvaluator()
metrics = evaluator.compute_all_metrics(predictions, targets)
print(f"BLEU-4: {metrics['bleu_4']:.3f}")
print(f"ROUGE-L: {metrics['rouge_l_f']:.3f}")
```

## Inference

```python
import torch
from src.models.eeg_model import EEGDecodingModel

# Load trained model
model = EEGDecodingModel.from_pretrained('path/to/checkpoint')
model.eval()

# Generate text from EEG
with torch.no_grad():
    generated_ids = model.generate(
        eeg_data=eeg_batch,
        max_length=20,
        num_beams=4
    )
    
text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Decoded text: {text}")
```

## Repository Structure

```
inner-speech-translation/
├── src/
│   ├── models/          # Neural network architectures
│   ├── data/            # Dataset and preprocessing
│   ├── training/        # Training loop and utilities
│   ├── evaluation/      # Metrics and evaluation
│   └── utils/           # Helper functions
├── scripts/             # Training and evaluation scripts
├── config/              # Configuration files
├── tests/               # Unit tests
└── notebooks/           # Analysis and visualization
```

## Performance

The model achieves competitive performance on EEG-to-Chinese text tasks:

- **BLEU-4**: ~15-25 (dataset dependent)
- **ROUGE-L**: ~20-35 (dataset dependent)
- **Training time**: ~2-4 hours on RTX 3090

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

```bibtex
@misc{eeg-to-text-2024,
  title={EEG-to-Text Neural Decoding for Chinese},
  author={Alex Steiner},
  year={2025},
  url={https://github.com/your-username/eeg-to-text}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Chinese BART model from [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese)
- Inspired by recent work in neural decoding