"""Compare checkpoint IoU at training and submission resolutions on labeled validation data."""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm

from src.data import PetDataset, build_transform
from src.predict import load_model, predict_probabilities


@torch.inference_mode()
def evaluate(model, root, input_size, score_size, device, batch_size, workers, limit, threshold):
    # Build 512-sized RGB images before submission-style inference. Score masks
    # are resized directly from original annotations with nearest interpolation.
    dataset = PetDataset(root, "test", build_transform((score_size, score_size)),
                         mask_size=(score_size, score_size))
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    metric = MeanIoU(num_classes=2, input_format="index", per_class=True).to(device)
    for images, masks in tqdm(loader, desc=f"input={input_size}, score={score_size}"):
        probabilities = predict_probabilities(
            model, images.to(device), input_size=(input_size, input_size)
        )
        metric.update((probabilities > threshold).long(), masks.to(device).long())
    background, pet = metric.compute().cpu().tolist()
    return {"input_size": input_size, "score_size": score_size, "images": len(dataset),
            "background_iou": background, "pet_iou": pet, "mean_iou": (background + pet) / 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N validation images (quick check)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive; workers must be nonnegative")
    if not 0 < args.threshold < 1:
        parser.error("threshold must be between 0 and 1")
    run = args.checkpoint.parent.parent
    cfg = OmegaConf.load(run / ".hydra/config.yaml")
    root = args.data_root or cfg.data.root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint).to(device).eval()
    results = []
    for input_size, score_size in [(256, 256), (256, 512), (512, 512)]:
        result = evaluate(model, root, input_size, score_size, device, args.batch_size,
                          args.workers, args.limit, args.threshold)
        results.append(result)
        print(f"Input {input_size}, mask {score_size}: pet IoU={result['pet_iou']:.5f}, "
              f"background IoU={result['background_iou']:.5f}, mean IoU={result['mean_iou']:.5f}", flush=True)
    output = args.output or run / f"{args.checkpoint.stem}_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "threshold": args.threshold,
                                  "split": "test", "data_root": str(root), "results": results}, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
