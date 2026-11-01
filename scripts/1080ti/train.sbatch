#!/bin/bash
#SBATCH --job-name=baseline-1080ti
#SBATCH --partition=gpu_devel
#SBATCH --qos=interactive
#SBATCH --gres=gpu:geforce:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd ~/cu_segmentation_comp
source .venv/bin/activate

nvidia-smi
python -u -m src.train \
    experiment=001_baseline \
    trainer.accelerator=cuda \
    trainer.devices=1 \
    trainer.precision=32-true \
    data.batch_size=4 \
    data.train_workers=3 \
    data.valid_workers=3 \
    "$@"
