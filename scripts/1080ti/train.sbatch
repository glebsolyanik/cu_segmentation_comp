#!/bin/bash -l
#SBATCH --job-name=baseline-1080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_devel
#SBATCH --qos=interactive
#SBATCH --gres=gpu:geforce:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd ~/cu_segmentation_comp
module load lib/openssl/3.5.0
source .venv/bin/activate

nvidia-smi
srun python -u -m src.train \
    experiment=001_baseline \
    trainer.accelerator=cuda \
    trainer.devices=1 \
    trainer.precision=32-true \
    data.batch_size=4 \
    data.train_workers=3 \
    data.valid_workers=3 \
    "hydra.run.dir=outputs/slurm_${SLURM_JOB_ID}" \
    "$@"
