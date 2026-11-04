"""Compare validation thresholds, horizontal-flip TTA and input resolutions."""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm

from src.data import PetDataset, build_transform
from src.predict import load_model, predict_probabilities


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module, dataloader: DataLoader, device: str,
    thresholds: list[float], score_size: tuple[int, int], hflip_tta: bool = False,
) -> list[dict]:
    metrics = {threshold: MeanIoU(num_classes=2, input_format="index").to(device) for threshold in thresholds}
    model.eval()
    for images, masks in tqdm(dataloader):
        probabilities = predict_probabilities(model, images.to(device), hflip_tta)
        if probabilities.shape[-2:] != score_size:
            probabilities = F.interpolate(
                probabilities.unsqueeze(1), size=score_size,
                mode="bilinear", align_corners=False,
            ).squeeze(1)
        targets = masks.to(device).long()
        for threshold, metric in metrics.items():
            metric.update((probabilities > threshold).long(), targets)
    return [{"threshold": threshold, "iou": metric.compute().item()} for threshold, metric in metrics.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--hflip-tta", action="store_true")
    parser.add_argument("--input-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional CSV of validation scores")
    args = parser.parse_args()
    if any(not 0 < threshold < 1 for threshold in args.thresholds):
        parser.error("Thresholds must be between 0 and 1")
    if args.input_size and any(size <= 0 for size in args.input_size):
        parser.error("Input dimensions must be positive")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Batch size must be positive and workers nonnegative")

    cfg = OmegaConf.load(args.checkpoint.parent.parent / ".hydra" / "config.yaml")
    input_size = tuple(args.input_size or cfg.data.image_size)
    score_size = tuple(OmegaConf.select(cfg, "evaluation.image_size", default=[256, 256]))
    dataset = PetDataset(
        args.data_root or cfg.data.root, "test", build_transform(input_size), mask_size=score_size
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint).to(device)
    scores = evaluate(model, loader, device, args.thresholds, score_size, args.hflip_tta)
    rows = []
    for score in scores:
        row = {
            "checkpoint": str(args.checkpoint), "input_size": f"{input_size[0]}x{input_size[1]}",
            "score_size": f"{score_size[0]}x{score_size[1]}", "hflip_tta": args.hflip_tta, **score,
        }
        rows.append(row)
        print(f"threshold={score['threshold']:.3f} valid_iou={score['iou']:.5f}")
    best = max(scores, key=lambda row: row["iou"])
    print(f"Best validation threshold: {best['threshold']:.3f} (IoU={best['iou']:.5f})")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
