#!/bin/bash
#SBATCH --job-name=esrgan-bc-1080ti
#SBATCH --partition=gpu_devel
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

cd ~/cu_segmentation_comp
source .venv/bin/activate

nvidia-smi
# gpu_devel nodes: GTX 1080 Ti (11 GB) and 4 CPU -> batch 4, 3 data workers
uv run python -m src.train experiment=001_baseline