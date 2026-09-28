#!/usr/bin/env python3
"""
AST (Audio Spectrogram Transformer) Training & Evaluation Pipeline
Converted from ast-model.ipynb for HPC and tmux environments.

Features:
- Full CLI argument parsing (override any hyperparameter or dataset path)
- Multi-worker DataLoader support with pin_memory for high GPU throughput on HPC
- Mixed-precision training (AMP) with automatic CUDA detection
- Fixed bug in layer freezing regex for HuggingFace AST encoder
- Fixed indentation bug in recording-level temporal mean pooling evaluation
- Prevents GPU VRAM accumulation during evaluation
- Comprehensive logging to both stdout and train.log file
- Saves best checkpoints for Stage 1, Stage 2, and species mapping JSON
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ASTForAudioClassification


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class Config:
    # Audio specs
    sample_rate: int = 32000
    clip_duration: float = 3.0  # 3 seconds clip
    clip_samples: int = field(init=False)

    # STFT & Mel FilterBank
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 50.0
    f_max: float = 14000.0

    # PCEN Hyperparameters (Per-Channel Energy Normalization)
    pcen_s: float = 0.025      # IIR filter smoothing
    pcen_alpha: float = 0.98   # AGC divisor exponent
    pcen_delta: float = 2.0    # Offset
    pcen_r: float = 0.5        # Root compression exponent
    pcen_eps: float = 1e-6

    # Augmentations
    freq_mask_param: int = 26  # Max mel channels masked
    time_mask_param: int = 64  # Max time frames masked
    num_freq_masks: int = 2
    num_time_masks: int = 2
    mixup_alpha: float = 0.4
    mixup_prob: float = 0.5

    # AST Model Backbone
    pretrained_ast_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ast_target_frames: int = 1024
    num_classes: int = 10      # Dynamically updated from dataset

    # 2-Stage Training Schedule
    stage1_lr: float = 5e-4
    stage1_freeze_layers: int = 8
    stage1_epochs: int = 25

    stage2_lr: float = 3e-5
    stage2_freeze_layers: int = 0
    stage2_epochs: int = 20
    stage2_dropout: float = 0.5
    stage2_weight_decay: float = 1e-2

    # Hardware & Performance
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    patience: int = 5

    def __post_init__(self):
        self.clip_samples = int(self.clip_duration * self.sample_rate)


# ==============================================================================
# Front-End: Mel-Spectrogram & PCEN
# ==============================================================================

class TorchMelPCEN(nn.Module):
    """
    Differentiable/deterministic PyTorch Mel-Spectrogram + PCEN front-end.
    Executes entirely on GPU with no Python overhead.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.s = config.pcen_s
        self.alpha = config.pcen_alpha
        self.delta = config.pcen_delta
        self.r = config.pcen_r
        self.eps = config.pcen_eps

        self.register_buffer("window", torch.hann_window(self.n_fft))

        mel_fb = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max,
        )
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb).float())

    def compute_stft(self, waveforms: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )

    def compute_power(self, stft: torch.Tensor) -> torch.Tensor:
        return stft.abs().pow(2)

    def compute_mel(self, power_spec: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.mel_fb, power_spec)

    def compute_baseline(self, mel_power: torch.Tensor) -> torch.Tensor:
        M = torch.empty_like(mel_power)
        M[:, :, 0] = self.s * mel_power[:, :, 0]
        for t in range(1, mel_power.shape[-1]):
            M[:, :, t] = (1.0 - self.s) * M[:, :, t - 1] + self.s * mel_power[:, :, t]
        return M

    def compute_pcen(self, mel_power: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        agc_denom = (self.eps + baseline).pow(self.alpha)
        return (mel_power / agc_denom + self.delta).pow(self.r) - (self.delta ** self.r)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        stft = self.compute_stft(waveforms)
        power_spec = self.compute_power(stft)
        mel_power = self.compute_mel(power_spec)
        baseline = self.compute_baseline(mel_power)
        pcen = self.compute_pcen(mel_power, baseline)
        return pcen.unsqueeze(1)  # (Batch, 1, n_mels, time_frames)


# ==============================================================================
# Data Augmentation: SpecAugment
# ==============================================================================

class TorchSpecAugment(nn.Module):
    """Time and Frequency Masking for spectrograms."""
    def __init__(self, config: Config):
        super().__init__()
        self.F = config.freq_mask_param
        self.T = config.time_mask_param
        self.nF = config.num_freq_masks
        self.nT = config.num_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        spec = spec.clone()
        _, _, n_mels, n_frames = spec.shape

        # Mask frequency bands
        for _ in range(self.nF):
            f = random.randint(0, self.F)
            f0 = random.randint(0, max(0, n_mels - f))
            spec[:, :, f0 : f0 + f, :] = 0.0

        # Mask time slices
        for _ in range(self.nT):
            t = random.randint(0, self.T)
            t0 = random.randint(0, max(0, n_frames - t))
            spec[:, :, :, t0 : t0 + t] = 0.0

        return spec


# ==============================================================================
# Dataset & Mixup Collator
# ==============================================================================

class BirdAudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        config: Config,
        species_to_idx: Dict[str, int],
        audio_root: Optional[str] = None,
    ):
        self.split = split
        self.config = config
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}
        self.num_classes = len(species_to_idx)

        # Audio root re-anchoring if specified
        if audio_root is not None:
            root_path = Path(audio_root).resolve()
            def fix_path(old_path: str) -> str:
                old = Path(old_path)
                if old.exists():
                    return str(old)
                parts = old.parts
                if "clips" in parts:
                    idx = parts.index("clips")
                    candidate = root_path / Path(*parts[idx:])
                    if candidate.exists():
                        return str(candidate)
                direct = root_path / "clips" / old.name
                if direct.exists():
                    return str(direct)
                return str(old)

            self.df["clip_path"] = self.df["clip_path"].apply(fix_path)

        # Class counts for loss re-weighting
        self.class_counts = np.zeros(self.num_classes, dtype=np.int64)
        for species, grp in self.df.groupby("species"):
            if species in self.species_to_idx:
                idx = self.species_to_idx[species]
                self.class_counts[idx] = len(grp)

    def _load_and_pad(self, audio_path: str) -> np.ndarray:
        try:
            audio, _ = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)  # Convert stereo to mono
        except Exception:
            audio = np.zeros(self.config.clip_samples, dtype=np.float32)

        if len(audio) < self.config.clip_samples:
            audio = np.pad(audio, (0, self.config.clip_samples - len(audio)))
        elif len(audio) > self.config.clip_samples:
            audio = audio[: self.config.clip_samples]
        return audio

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        waveform = self._load_and_pad(row["clip_path"])
        label = self.species_to_idx[row["species"]]
        rec_id = str(row.get("recording_id", idx))

        return {
            "waveform": torch.from_numpy(waveform),
            "label": torch.tensor(label, dtype=torch.long),
            "recording_id": rec_id,
        }


class MixupCollator:
    """Applies acoustic Mixup on waveform batches for regularization."""
    def __init__(self, num_classes: int, alpha: float = 0.4, prob: float = 0.5):
        self.num_classes = num_classes
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch: List[dict]) -> dict:
        waveforms = torch.stack([b["waveform"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        recording_ids = [b["recording_id"] for b in batch]

        one_hot = torch.zeros(len(batch), self.num_classes, dtype=torch.float32)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        if np.random.rand() > self.prob or len(batch) <= 1:
            return {"waveform": waveforms, "label": one_hot, "recording_id": recording_ids}

        perm = torch.randperm(len(batch))
        lam = float(np.random.beta(self.alpha, self.alpha))

        mixed_waveforms = lam * waveforms + (1.0 - lam) * waveforms[perm]
        mixed_labels = lam * one_hot + (1.0 - lam) * one_hot[perm]

        return {"waveform": mixed_waveforms, "label": mixed_labels, "recording_id": recording_ids}


# ==============================================================================
# Model Architecture
# ==============================================================================

class ASTBirdClassifier(nn.Module):
    """Audio Spectrogram Transformer classifier with configurable layer freezing."""
    def __init__(
        self,
        num_classes: int,
        config: Config,
        dropout: float = 0.0,
        freeze_layers: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.target_time_frames = config.ast_target_frames

        # Load AudioSet-pretrained backbone
        self.ast = ASTForAudioClassification.from_pretrained(
            config.pretrained_ast_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # Gradient checkpointing for reduced VRAM usage on HPC GPUs
        self.ast.gradient_checkpointing_enable()

        # Regularization dropout before classification head
        if dropout > 0.0:
            orig_dense = self.ast.classifier.dense
            self.ast.classifier.dense = nn.Sequential(nn.Dropout(p=dropout), orig_dense)

        # Freeze bottom layers if requested
        if freeze_layers > 0:
            self.freeze_bottom_layers(freeze_layers)

    def freeze_bottom_layers(self, n_layers: int):
        """
        Freezes patch embeddings and the first n_layers of encoder transformer blocks.
        Fixed regex: matches 'encoder.layer.<idx>' in HuggingFace AST.
        """
        frozen_count = 0
        total_count = 0
        for name, param in self.ast.named_parameters():
            total_count += 1
            if "embeddings" in name:
                param.requires_grad = False
                frozen_count += 1
            elif "encoder" in name:
                match = re.search(r"layers?[._](\d+)", name)
                if match and int(match.group(1)) < n_layers:
                    param.requires_grad = False
                    frozen_count += 1
        logging.info(f"Froze {frozen_count}/{total_count} parameter tensors (layers 0 to {n_layers-1} + embeddings)")

    def unfreeze_all_layers(self):
        """Unfreezes all parameters for full end-to-end fine-tuning."""
        for param in self.ast.parameters():
            param.requires_grad = True
        logging.info("Unfroze all model parameters for full fine-tuning.")

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Input:  spec of shape (Batch, 1, 128, Time_Frames)
        Output: logits of shape (Batch, num_classes)
        """
        x = spec.squeeze(1).transpose(1, 2)  # (Batch, Time_Frames, 128)
        t_frames = x.shape[1]
        if t_frames < self.target_time_frames:
            pad = self.target_time_frames - t_frames
            x = F.pad(x, (0, 0, 0, pad))
        elif t_frames > self.target_time_frames:
            x = x[:, : self.target_time_frames, :]

        outputs = self.ast(input_values=x)
        return outputs.logits


# ==============================================================================
# Loss & Metrics
# ==============================================================================

class ClassBalancedBCELoss(nn.Module):
    """Class-balanced BCE loss to handle species imbalance."""
    def __init__(self, class_counts: Optional[np.ndarray] = None, beta: float = 0.999):
        super().__init__()
        pos_weight = None
        if class_counts is not None and len(class_counts) > 0:
            counts = np.maximum(class_counts, 1)
            effective_num = 1.0 - np.power(beta, counts)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.mean(weights)
            weights = np.clip(weights, 1.0, 25.0)
            pos_weight = torch.tensor(weights, dtype=torch.float32)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 1:
            targets = F.one_hot(targets, num_classes=logits.shape[-1]).float()
        return self.loss_fn(logits, targets)


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """Computes Top-1 and Top-5 accuracy."""
    with torch.no_grad():
        target_indices = targets.argmax(dim=1) if targets.ndim > 1 else targets
        top_k = min(5, logits.shape[1])
        _, topk_preds = logits.topk(top_k, dim=1)
        top1_correct = (topk_preds[:, 0] == target_indices).sum().item()
        top5_correct = (topk_preds == target_indices.unsqueeze(1)).any(dim=1).sum().item()
        n = max(len(targets), 1)
        return top1_correct / n, top5_correct / n


# ==============================================================================
# Training Stage Runner
# ==============================================================================

def run_training_stage(
    stage_num: int,
    epochs: int,
    lr: float,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    mel_pcen: nn.Module,
    spec_aug: nn.Module,
    output_dir: Path,
    patience: int = 5,
    device: str = "cuda",
) -> float:
    logging.info(f"{'='*20} STARTING STAGE {stage_num} ({epochs} Epochs | LR={lr}) {'='*20}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-2 if stage_num == 2 else 1e-3,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_cuda = "cuda" in str(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_val_top1 = 0.0
    patience_counter = 0
    best_checkpoint_path = output_dir / f"best_stage{stage_num}_model.pt"

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss, train_top1, total_samples = 0.0, 0.0, 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]",
            leave=False,
            file=sys.stdout,
        )
        for batch in pbar:
            waveforms = batch["waveform"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.no_grad():
                specs = mel_pcen(waveforms)
                specs = spec_aug(specs)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(specs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_top1, _ = compute_metrics(logits, labels)
            bs = len(labels)
            train_loss += loss.item() * bs
            train_top1 += batch_top1 * bs
            total_samples += bs

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(train_top1 / total_samples)*100:.1f}%",
            })

        train_loss /= max(total_samples, 1)
        train_top1 /= max(total_samples, 1)

        # 2. Validation Phase
        model.eval()
        val_loss, val_top1, val_top5, total_val = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                waveforms = batch["waveform"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    specs = mel_pcen(waveforms)
                    logits = model(specs)
                    loss = criterion(logits, labels)

                b_top1, b_top5 = compute_metrics(logits, labels)
                bs = len(labels)
                val_loss += loss.item() * bs
                val_top1 += b_top1 * bs
                val_top5 += b_top5 * bs
                total_val += bs

        val_loss /= max(total_val, 1)
        val_top1 /= max(total_val, 1)
        val_top5 /= max(total_val, 1)

        scheduler.step()

        msg = (
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_top1*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Top-1: {val_top1*100:.2f}% Top-5: {val_top5*100:.2f}%"
        )
        logging.info(msg)

        # 3. Checkpointing & Early Stopping
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            logging.info(f"  >>> Best Stage {stage_num} model saved to {best_checkpoint_path} (Val Top-1: {val_top1*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"  ⏹️ Early stopping triggered at epoch {epoch}. Best Val Top-1: {best_val_top1*100:.2f}%")
                break

    # Reload best weights for subsequent stage
    if best_checkpoint_path.exists():
        logging.info(f"Reloading best checkpoint from {best_checkpoint_path}")
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    return best_val_top1


# ==============================================================================
# Recording-Level Evaluation with Temporal Mean Pooling
# ==============================================================================

def evaluate_recording_level(
    model: nn.Module,
    loader: DataLoader,
    mel_pcen: nn.Module,
    device: str = "cuda",
    split_name: str = "Validation",
) -> Tuple[float, float]:
    """
    Evaluates whole recordings by aggregating probabilities across individual 3s clips
    using Temporal Mean Pooling.
    (Fixes the notebook loop indentation bug and CPU tensor offloading).
    """
    model.eval()
    rec_probs = defaultdict(list)
    rec_targets = {}
    use_cuda = "cuda" in str(device)

    logging.info(f"Evaluating {split_name} split across {len(loader.dataset)} clips...")

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval ({split_name})", leave=False, file=sys.stdout):
            waveforms = batch["waveform"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            rec_ids = batch["recording_id"]

            with torch.amp.autocast("cuda", enabled=use_cuda):
                specs = mel_pcen(waveforms)
                logits = model(specs)

            probs = torch.sigmoid(logits)

            # Move probabilities to CPU to prevent GPU VRAM memory leak
            probs_cpu = probs.cpu()
            labels_cpu = labels.cpu()

            for r_id, p, l in zip(rec_ids, probs_cpu, labels_cpu):
                rec_probs[r_id].append(p)
                if r_id not in rec_targets:
                    rec_targets[r_id] = l.argmax().item() if l.ndim > 0 else l.item()

    rec_top1 = 0
    rec_top5 = 0
    total_recs = len(rec_probs)

    if total_recs == 0:
        logging.warning("No recordings found for evaluation.")
        return 0.0, 0.0

    for r_id, p_list in rec_probs.items():
        avg_prob = torch.stack(p_list).mean(dim=0)
        target = rec_targets[r_id]
        top_k = min(5, len(avg_prob))
        _, top5_indices = avg_prob.topk(top_k)

        if top5_indices[0].item() == target:
            rec_top1 += 1
        if target in top5_indices.tolist():
            rec_top5 += 1

    top1_pct = (rec_top1 / total_recs) * 100.0
    top5_pct = (rec_top5 / total_recs) * 100.0

    sep = "=" * 60
    logging.info(f"\n{sep}")
    logging.info(f"RECORDING-LEVEL RESULTS ({split_name} - {total_recs} unique field recordings):")
    logging.info(f"  Top-1 Accuracy: {top1_pct:.2f}%")
    logging.info(f"  Top-5 Accuracy: {top5_pct:.2f}%")
    logging.info(f"{sep}\n")

    return top1_pct, top5_pct


# ==============================================================================
# Main Orchestration & CLI
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIPS_CSV = str(PROJECT_ROOT / "outputs" / "data" / "final" / "csv" / "clips_metadata.csv")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "models")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AST for Bird Audio Classification on HPC / tmux",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    parser.add_argument("--clips_csv", type=str, default=DEFAULT_CLIPS_CSV, help="Path to clips metadata CSV")
    parser.add_argument("--audio_root", type=str, default=None, help="Root folder for audio clips if relocated")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save checkpoints and logs")
    
    # Model
    parser.add_argument("--pretrained_ast", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593", help="Pretrained AST checkpoint")
    parser.add_argument("--dropout", type=float, default=0.5, help="Stage 2 dropout")
    
    # Training Stages
    parser.add_argument("--stage1_epochs", type=int, default=25, help="Stage 1 warmup epochs")
    parser.add_argument("--stage1_lr", type=float, default=5e-4, help="Stage 1 learning rate")
    parser.add_argument("--stage1_freeze", type=int, default=8, help="Number of bottom transformer layers to freeze in stage 1")
    
    parser.add_argument("--stage2_epochs", type=int, default=20, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--stage2_lr", type=float, default=3e-5, help="Stage 2 learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    
    # Batch & Hardware
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--device", type=str, default=None, help="Torch device: cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Execution modes
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only using an existing checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to evaluate (for --eval_only)")
    parser.add_argument("--eval_test_split", action="store_true", help="Also evaluate test split if present in metadata")
    
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info(f"Logging initialized. Output directory: {output_dir.resolve()}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    set_seed(args.seed)

    # Device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device} | CUDA Available: {torch.cuda.is_available()}")
    if "cuda" in device and torch.cuda.is_available():
        dev_idx = torch.cuda.current_device() if ":" not in device else int(device.split(":")[-1])
        logging.info(f"GPU Device Name: {torch.cuda.get_device_name(dev_idx)}")

    # Load Metadata
    clips_csv_path = Path(args.clips_csv)
    if not clips_csv_path.exists():
        logging.error(f"Metadata CSV not found at: {clips_csv_path.resolve()}")
        sys.exit(1)

    logging.info(f"Loading metadata from {clips_csv_path.resolve()}...")
    df = pd.read_csv(clips_csv_path)

    # Class mappings
    all_species = sorted(df["species"].unique().tolist())
    species_to_idx = {s: i for i, s in enumerate(all_species)}
    num_classes = len(all_species)
    logging.info(f"Dataset species count: {num_classes} classes")

    # Save species mapping JSON for inference later
    species_map_file = output_dir / "species_map.json"
    with open(species_map_file, "w", encoding="utf-8") as f:
        json.dump({"species_to_idx": species_to_idx, "idx_to_species": {str(i): s for i, s in enumerate(all_species)}}, f, indent=2)
    logging.info(f"Saved species map to {species_map_file}")

    # Build Config
    cfg = Config(
        num_classes=num_classes,
        pretrained_ast_name=args.pretrained_ast,
        stage1_lr=args.stage1_lr,
        stage1_freeze_layers=args.stage1_freeze,
        stage1_epochs=args.stage1_epochs,
        stage2_lr=args.stage2_lr,
        stage2_epochs=args.stage2_epochs,
        stage2_dropout=args.dropout,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
        patience=args.patience,
    )

    # Initialize Front-End
    mel_pcen = TorchMelPCEN(cfg).to(cfg.device)
    spec_aug = TorchSpecAugment(cfg).to(cfg.device)

    # If Evaluation-Only mode
    if args.eval_only:
        ckpt_path = args.checkpoint or str(output_dir / "best_stage2_model.pt")
        if not os.path.exists(ckpt_path):
            logging.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        logging.info(f"Evaluation-Only mode with checkpoint: {ckpt_path}")
        model = ASTBirdClassifier(num_classes=num_classes, config=cfg).to(cfg.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))

        val_dataset = BirdAudioDataset(df, split="val", config=cfg, species_to_idx=species_to_idx, audio_root=args.audio_root)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=("cuda" in str(cfg.device)),
        )
        evaluate_recording_level(model, val_loader, mel_pcen, device=cfg.device, split_name="Validation")

        if args.eval_test_split and "test" in df["split"].values:
            test_dataset = BirdAudioDataset(df, split="test", config=cfg, species_to_idx=species_to_idx, audio_root=args.audio_root)
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=("cuda" in str(cfg.device)),
            )
            evaluate_recording_level(model, test_loader, mel_pcen, device=cfg.device, split_name="Test")
        return

    # Datasets
    train_dataset = BirdAudioDataset(df, split="train", config=cfg, species_to_idx=species_to_idx, audio_root=args.audio_root)
    val_dataset = BirdAudioDataset(df, split="val", config=cfg, species_to_idx=species_to_idx, audio_root=args.audio_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=MixupCollator(num_classes=num_classes, alpha=cfg.mixup_alpha, prob=cfg.mixup_prob),
        num_workers=cfg.num_workers,
        pin_memory=("cuda" in str(cfg.device)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=("cuda" in str(cfg.device)),
    )

    logging.info(f"✓ Train clips count: {len(train_dataset)}")
    logging.info(f"✓ Val clips count:   {len(val_dataset)}")
    logging.info(f"✓ Number of classes: {num_classes}")

    criterion = ClassBalancedBCELoss(class_counts=train_dataset.class_counts).to(cfg.device)

    # 1. Initialize Model for Stage 1 (Frozen bottom layers)
    logging.info("Initializing AST Classifier for Stage 1...")
    model = ASTBirdClassifier(
        num_classes=num_classes,
        dropout=cfg.stage2_dropout,
        freeze_layers=cfg.stage1_freeze_layers,
        config=cfg,
    ).to(cfg.device)

    # 2. Stage 1: Warmup Training
    run_training_stage(
        stage_num=1,
        epochs=cfg.stage1_epochs,
        lr=cfg.stage1_lr,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        mel_pcen=mel_pcen,
        spec_aug=spec_aug,
        output_dir=output_dir,
        patience=cfg.patience,
        device=cfg.device,
    )

    # 3. Stage 2: Full Fine-Tuning
    logging.info("Transitioning to Stage 2: Unfreezing all backbone layers...")
    model.unfreeze_all_layers()

    run_training_stage(
        stage_num=2,
        epochs=cfg.stage2_epochs,
        lr=cfg.stage2_lr,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        mel_pcen=mel_pcen,
        spec_aug=spec_aug,
        output_dir=output_dir,
        patience=cfg.patience,
        device=cfg.device,
    )

    # 4. Final Evaluation (Validation Split)
    logging.info("Executing final recording-level temporal mean pooling evaluation...")
    evaluate_recording_level(model, val_loader, mel_pcen, device=cfg.device, split_name="Validation")

    # 5. Evaluate Test Split if available
    if args.eval_test_split or "test" in df["split"].values:
        test_dataset = BirdAudioDataset(df, split="test", config=cfg, species_to_idx=species_to_idx, audio_root=args.audio_root)
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=("cuda" in str(cfg.device)),
            )
            evaluate_recording_level(model, test_loader, mel_pcen, device=cfg.device, split_name="Test")

    logging.info("🎉 All training and evaluation stages completed successfully!")


if __name__ == "__main__":
    main()
