"""Visualize validation predictions or test masks from a submission CSV."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from src.data import PetDataset, build_transform
from src.predict import experiment_name, load_model
from src.rle import rle_decode


def make_panel(image: np.ndarray, target: np.ndarray | None, prediction: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    overlay[prediction] = (
        0.55 * image[prediction] + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)
    panels = [("Image", image)]
    if target is not None:
        panels.append(("Ground truth", np.repeat((target * 255)[..., None], 3, axis=2)))
    panels.extend([
        ("Prediction", np.repeat((prediction.astype(np.uint8) * 255)[..., None], 3, axis=2)),
        ("Overlay", overlay),
    ])
    labeled = []
    for title, panel in panels:
        panel = cv2.copyMakeBorder(panel, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        labeled.append(panel)
    return np.concatenate(labeled, axis=1)


def save_panel(panel: np.ndarray, path: Path) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to save {path}")
    print(path)


def visualize_submission(args) -> None:
    with args.submission.open(newline="") as file:
        rows = list(csv.DictReader(file))
    # Shuffle a stable ID order so increasing num-images keeps existing examples.
    rows.sort(key=lambda row: int(row["ImageID"]))
    indices = np.random.default_rng(args.seed).permutation(len(rows))[:args.num_images]
    output_dir = (args.output_dir or Path("predictions") / "test" /
                  experiment_name(args.submission) / args.submission.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        row = rows[int(idx)]
        image_id = int(row["ImageID"])
        image_path = args.test_dir / f"{image_id}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = rle_decode(row["RLE"], image.shape[:2])
        save_panel(make_panel(image, None, mask), output_dir / f"{image_id}.png")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--submission", type=Path, help="Visualize test masks from CSV instead of validation")
    parser.add_argument("--test-dir", type=Path, default=Path("data/test/test/images"))
    parser.add_argument("--data-root", help="Override the training data root stored in the run config")
    parser.add_argument("--output-dir", type=Path, help="Default: predictions/valid/<experiment> or predictions/test/<experiment>/<CSV name>")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.submission is None):
        parser.error("Provide either a checkpoint or --submission CSV")
    if args.num_images < 1:
        parser.error("--num-images must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    if args.submission is not None:
        visualize_submission(args)
        return

    run_dir = args.checkpoint.parent.parent
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    output_dir = args.output_dir or Path("predictions") / "valid" / str(cfg.name)
    dataset = PetDataset(
        args.data_root or cfg.data.root, "test", build_transform(tuple(cfg.data.image_size))
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint).to(device).eval()
    indices = np.random.default_rng(args.seed).choice(
        len(dataset), size=min(args.num_images, len(dataset)), replace=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        image, target = dataset[int(idx)]
        prediction = (model(image.unsqueeze(0).to(device))[0, 0].sigmoid() > args.threshold).cpu().numpy()
        rgb = (image.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        panel = make_panel(rgb, target.numpy().astype(np.uint8), prediction)
        path = output_dir / f"{dataset.image_names[int(idx)]}.png"
        save_panel(panel, path)


if __name__ == "__main__":
    main()
