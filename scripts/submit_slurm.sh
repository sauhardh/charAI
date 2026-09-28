#!/usr/bin/env bash
#SBATCH --job-name=ast_bird
#SBATCH --output=outputs/models/%x_%j.log
#SBATCH --error=outputs/models/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Move to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Load HPC environment modules if applicable
# module load python/3.10 cuda/12.1

mkdir -p outputs/models

# Activate virtual environment or conda if applicable
# source .venv/bin/activate
# conda activate charai

echo "Starting job on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python3 hpc/train.py \
    --clips_csv outputs/data/final/csv/clips_metadata.csv \
    --output_dir outputs/models \
    --batch_size 16 \
    --num_workers 6 \
    --stage1_epochs 25 \
    --stage2_epochs 20
