# HPC Training & Execution Guide

This folder contains scripts and utilities for running the Audio Spectrogram Transformer (AST) training pipeline on High Performance Computing (HPC) clusters using `tmux` or the Slurm workload manager.

---

## Files in this Directory

| File | Description |
| :--- | :--- |
| `train.py` | Full AST training and evaluation script with multi-worker loading, AMP, and CLI flags |
| `run_tmux.sh` | One-command launcher to start training inside a detached `tmux` session |
| `submit_slurm.sh` | Slurm batch submission script template (`sbatch submit_slurm.sh`) |
| `requirements.txt` | Minimal Python dependencies for cluster environments |

---

## 1. Quick Start: Running in `tmux`

From the project root:

```bash
# Launch training in a detached tmux session named "ast_train"
./hpc/run_tmux.sh
```

### Essential tmux Commands

- **Attach to session**:
  ```bash
  tmux attach -t ast_train
  ```
- **Detach from session**:
  Press `Ctrl + b`, then release both and press `d`. Training continues running in the background even if you disconnect from SSH.
- **Inspect live training log**:
  ```bash
  tail -f outputs/models/train.log
  ```
- **Monitor GPU utilization**:
  ```bash
  watch -n 1 nvidia-smi
  ```
- **Terminate the session (if needed)**:
  ```bash
  tmux kill-session -t ast_train
  ```

---

## 2. Running Manually with Custom Arguments

You can run `train.py` directly with custom hyperparameters:

```bash
python3 hpc/train.py \
  --batch_size 32 \
  --num_workers 6 \
  --stage1_epochs 30 \
  --stage2_epochs 25 \
  --device cuda:0
```

### Common CLI Options

- `--clips_csv`: Path to CSV metadata (default: `outputs/data/final/csv/clips_metadata.csv`)
- `--audio_root`: Path to root directory of audio clips if moved on HPC
- `--output_dir`: Directory for checkpoints and logs (default: `outputs/models`)
- `--eval_only`: Skip training and evaluate a checkpoint
- `--checkpoint`: Path to `.pt` checkpoint to evaluate
- `--eval_test_split`: Also run evaluation on test set

---

## 3. Submitting via Slurm Batch Job

If your HPC cluster uses Slurm:

```bash
sbatch hpc/submit_slurm.sh
```

Check job status:
```bash
squeue -u $USER
```
