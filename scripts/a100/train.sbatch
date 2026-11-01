#!/bin/bash -l
#SBATCH --job-name=baseline-a100
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd ~/cu_segmentation_comp
module load lib/openssl/3.5.0
source .venv/bin/activate

export NO_ALBUMENTATIONS_UPDATE=1
export HYDRA_FULL_ERROR=1
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"

nvidia-smi
python -u -m src.train \
    experiment=001_baseline \
    trainer.accelerator=cuda \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    data.batch_size=32 \
    data.train_workers=6 \
    data.valid_workers=6 \
    "$@"
