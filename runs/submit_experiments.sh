#!/usr/bin/env bash
# Usage: bash runs/submit_experiments.sh 1080ti [experiment ...]
# Run from the repository root on the cluster after uv sync.
set -euo pipefail

profile="${1:-1080ti}"
if [[ $# -gt 0 ]]; then shift; fi
case "$profile" in
    1080ti) launcher="runs/1080ti/train_2.sbatch" ;;
    a100) launcher="runs/a100/train.sbatch" ;;
    *) echo "Unknown profile: $profile (choose 1080ti or a100)" >&2; exit 1 ;;
esac

if [[ $# -eq 0 ]]; then
    set -- 002_augmentations 003_bce_dice 004_cosine \
        005_unet_resnet18_scratch 006_unet_resnet18_pretrained \
        007_resolution384 008_segformer
fi

# Validate all names before submitting any job.
for experiment in "$@"; do
    if [[ ! -f "configs/experiment/${experiment}.yaml" ]]; then
        echo "Missing experiment config: $experiment" >&2
        exit 1
    fi
done
mkdir -p logs
for experiment in "$@"; do
    sbatch --job-name="${experiment}-${profile}" "$launcher" "experiment=$experiment"
done
