# CharAI: Bird Species Audio Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-AST-yellow.svg)](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**CharAI** is a bioacoustic machine learning pipeline for bird species classification from field audio recordings. It leverages **Audio Spectrogram Transformers (AST)** combined with a **Per-Channel Energy Normalization (PCEN)** front-end and a 2-stage fine-tuning curriculum to handle complex, noisy outdoor acoustic environments.

---

## Architecture & Methodological Highlights

- **Audio Front-End**: 
  - Standardized to 32 kHz mono audio.
  - Computes STFT (2048 FFT, 512 hop) followed by 128 Mel filterbanks (50 Hz – 14 kHz).
  - **PCEN (Per-Channel Energy Normalization)**: Suppresses stationary background noise and dynamic range variations better than log-mel compression.
- **Regularization & Augmentations**:
  - **SpecAugment**: Multi-band frequency and temporal masking.
  - **Acoustic Mixup**: Regularizes label ambiguity and multi-species background calls.
- **Model Backbone**:
  - Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`) pre-trained on AudioSet.
  - Gradient checkpointing enabled for low VRAM consumption during training.
- **2-Stage Fine-Tuning Strategy**:
  - **Stage 1 (Warmup)**: Freezes patch embeddings and bottom 8 transformer encoder blocks, training higher layers with a higher learning rate ($5 \times 10^{-4}$).
  - **Stage 2 (End-to-End)**: Unfreezes all backbone layers for full fine-tuning with Cosine Annealing ($3 \times 10^{-5}$) and early stopping.
- **Class-Balanced BCE Loss**: Addresses species call frequency imbalance in field data.
- **Recording-Level Benchmark**: Aggregates clip predictions per continuous field recording via **Temporal Mean Pooling** for Top-1 and Top-5 accuracy.

---

## Repository Structure

```text
charai/
├── preprocessing/                 # Data collection and audio preparation
│   ├── download_metadata.py       # Queries Xeno-Canto API for species recordings
│   ├── filter_split_metadata.py   # Stratified splitting (train/val/test) & filtering
│   ├── download_audio.py          # Parallel audio downloader from Xeno-Canto
│   ├── normalize_audio.py         # Resamples audio to 32 kHz and converts to mono
│   └── extract_clip.py            # Extracts uniform 3-second audio clips
│
├── scripts/                       # Model training and cluster execution
│   ├── train.py                   # Standalone AST training & evaluation pipeline
│   ├── run_tmux.sh                # Launcher for persistent background runs in tmux
│   ├── submit_slurm.sh            # Slurm cluster batch job script (sbatch)
│   ├── requirements.txt           # Minimal pip dependencies
│   └── README.md                  # HPC & cluster execution guide
│
├── outputs/                       # Generated data, models, and artifacts
│   ├── data/
│   │   ├── initial/               # Raw downloaded audio and metadata
│   │   └── final/
│   │       ├── clips/             # 3-second processed audio clips
│   │       └── csv/               # clips_metadata.csv
│   └── models/                    # Saved checkpoints (.pt) and train.log
│
├── ast-model.ipynb                # Interactive research notebook
├── pyproject.toml                 # uv / pip project configuration
└── README.md                      # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.10+ (or Python 3.13)
- CUDA-compatible GPU (recommended for training; CPU supported for dry runs)

### Setup Environment

Using `uv`:
```bash
uv sync
source .venv/bin/activate
```

Or using standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## End-to-End Workflow

### 1. Data Preprocessing

Run the preprocessing steps to fetch metadata, download recordings from Xeno-Canto, normalize audio, and extract 3-second training clips:

```bash
# 1. Fetch metadata from Xeno-Canto API
python3 preprocessing/download_metadata.py

# 2. Filter species and create train/val/test splits
python3 preprocessing/filter_split_metadata.py

# 3. Download audio recordings
python3 preprocessing/download_audio.py

# 4. Normalize sample rates and format
python3 preprocessing/normalize_audio.py

# 5. Extract uniform 3-second audio clips
python3 preprocessing/extract_clip.py
```

This generates `outputs/data/final/csv/clips_metadata.csv` and audio clips under `outputs/data/final/clips/`.

---

### 2. Model Training

#### Option A: Local Training (Direct CLI)
```bash
python3 scripts/train.py \
  --batch_size 16 \
  --num_workers 4 \
  --stage1_epochs 25 \
  --stage2_epochs 20
```

#### Option B: Background Run in `tmux` (Laptop or HPC)
Keeps training alive in the background even if you disconnect or close the terminal:
```bash
./scripts/run_tmux.sh
```
- **Attach to session**: `tmux attach -t ast_train`
- **Detach from session**: Press `Ctrl + b`, then release and press `d`
- **Stream live logs**: `tail -f outputs/models/train.log`
- **Kill session**: `tmux kill-session -t ast_train`

#### Option C: Slurm Cluster Submission
If running on an HPC cluster with Slurm:
```bash
sbatch scripts/submit_slurm.sh
```

---

## Checkpoints & Evaluation

During training, best models and species index maps are automatically saved to `outputs/models/`:
- `best_stage1_model.pt`: Checkpoint from Stage 1 warmup
- `best_stage2_model.pt`: Best fine-tuned model checkpoint
- `species_map.json`: Mapping between class IDs and species names
- `train.log`: Full timestamped training and validation logs

### Evaluate an Existing Checkpoint
```bash
python3 scripts/train.py \
  --eval_only \
  --checkpoint outputs/models/best_stage2_model.pt \
  --eval_test_split
```

---

## CLI Options Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--clips_csv` | `outputs/data/final/csv/clips_metadata.csv` | Path to dataset metadata CSV |
| `--audio_root` | `None` | Custom audio clips folder (if moved on cluster) |
| `--output_dir` | `outputs/models` | Output directory for checkpoints and logs |
| `--batch_size` | `16` | Mini-batch size |
| `--num_workers` | `4` | DataLoader parallel worker processes |
| `--device` | `cuda` (if available) else `cpu` | Device (`cuda`, `cuda:0`, `cpu`) |
| `--stage1_epochs`| `25` | Stage 1 warmup epochs |
| `--stage1_lr` | `5e-4` | Stage 1 learning rate |
| `--stage1_freeze`| `8` | Number of bottom transformer layers frozen |
| `--stage2_epochs`| `20` | Stage 2 full fine-tuning epochs |
| `--stage2_lr` | `3e-5` | Stage 2 learning rate |
| `--dropout` | `0.5` | Classification head dropout rate |
| `--patience` | `5` | Early stopping patience (epochs) |
| `--seed` | `42` | Random seed for reproducibility |
| `--eval_only` | `False` | Run recording-level evaluation only |
| `--eval_test_split` | `False` | Evaluate on test split in addition to validation |
