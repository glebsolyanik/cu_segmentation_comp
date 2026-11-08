"""Show selected external examples and their binary pet masks."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/external/coco_people_train2017"))
    parser.add_argument("--output-dir", type=Path, default=Path("predictions/dataset/coco_people_train2017"))
    parser.add_argument("--per-group", type=int, default=12)
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error("per-group must be positive")
    manifest = json.loads((args.data_root / "manifest.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual = args.output_dir / "images"
    individual.mkdir(exist_ok=True)
    for sample in manifest["samples"]:
        image = cv2.imread(str(args.data_root / sample["image"]))
        mask = cv2.imread(str(args.data_root / sample["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise OSError(sample["name"])
        height, width = image.shape[:2]
        # Keep the original aspect ratio; cap panel width for comfortable viewing.
        display_width = min(width, 640)
        display_height = max(1, round(height * display_width / width))
        image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (display_width, display_height), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = image.copy()
        overlay[mask] = (image[mask] * 0.55 + np.array([0, 255, 0]) * 0.45).astype(np.uint8)
        binary = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
        panels = []
        for title, view in [("Original", image), ("Ground truth", binary), ("Overlay", overlay)]:
            view = cv2.copyMakeBorder(view, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(view, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(view)
        grid = np.concatenate(panels, axis=1)
        if not cv2.imwrite(str(individual / f"{sample['name']}.jpg"), grid):
            raise OSError(sample["name"])
    if manifest.get("training_allowed") is False:
        print("WARNING: this dataset is for inspection only; excluded from training.")
    rng = np.random.default_rng(42)
    for group in sorted({sample["group"] for sample in manifest["samples"]}):
        pool = [sample for sample in manifest["samples"] if sample["group"] == group]
        selected = [pool[int(index)] for index in rng.permutation(len(pool))[:args.per_group]]
        size, columns = 256, 3
        canvas = np.full((math.ceil(len(selected) / columns) * (size + 28), columns * size * 2, 3), 255, np.uint8)
        for index, sample in enumerate(selected):
            image = cv2.imread(str(args.data_root / sample["image"]))
            mask = cv2.imread(str(args.data_root / sample["mask"]), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise OSError(sample["name"])
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay = image.copy()
            overlay[mask] = (image[mask] * 0.55 + np.array([0, 255, 0]) * 0.45).astype(np.uint8)
            row, col = divmod(index, columns)
            y, x = row * (size + 28), col * 2 * size
            cv2.putText(canvas, f"{group} - {sample['coco_image_id']}", (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            canvas[y + 28:y + 28 + size, x:x + size] = image
            canvas[y + 28:y + 28 + size, x + size:x + 2 * size] = overlay
        path = args.output_dir / f"{group}.jpg"
        if not cv2.imwrite(str(path), canvas):
            raise OSError(path)
        print(path)


if __name__ == "__main__":
    main()
