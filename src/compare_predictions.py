"""Create one labeled comparison grid per test image from experiment submissions."""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.rle import rle_decode


def read_submission(path: Path) -> dict[int, str]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if not {"ImageID", "RLE"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing ImageID/RLE columns: {path}")
        result = {}
        for row in reader:
            image_id = int(row["ImageID"])
            if image_id in result:
                raise ValueError(f"Duplicate ImageID {image_id}: {path}")
            result[image_id] = row["RLE"]
    return result


def labeled_tile(image: np.ndarray, label: str, size: int, mask: bool = False) -> np.ndarray:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = cv2.getTextSize(label, font, 0.6, 1)[0][0]
    scale = min(0.6, 0.6 * (size - 16) / max(width, 1))
    cv2.putText(image, label, (8, 24), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions/test"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test/test/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("predictions/comparisons/test"))
    parser.add_argument("--submission-name", default="submission.csv")
    parser.add_argument("--experiments", nargs="+", help="Optional experiment folder names, in display order")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--mode", choices=("overlay", "mask"), default="overlay")
    args = parser.parse_args()
    if args.columns < 1 or args.tile_size < 64:
        parser.error("columns must be positive and tile-size must be at least 64")
    folders = ([args.predictions_dir / name for name in args.experiments] if args.experiments
               else sorted(path.parent for path in args.predictions_dir.glob(f"*/{args.submission_name}")))
    if not folders:
        parser.error("No experiment submissions found")
    submissions = [(folder.name, read_submission(folder / args.submission_name)) for folder in folders]
    images = sorted(args.test_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not images:
        parser.error("No test PNG images found")
    ids = {int(path.stem) for path in images}
    for name, masks in submissions:
        missing = ids - masks.keys()
        if missing:
            parser.error(f"{name}: missing masks for {len(missing)} images, e.g. {sorted(missing)[:5]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Experiments: " + ", ".join(name for name, _ in submissions), flush=True)
    columns = min(args.columns, len(submissions) + 1)
    rows = math.ceil((len(submissions) + 1) / columns)
    for path in tqdm(images, desc="Comparison grids"):
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"Cannot read {path}")
        image_id = int(path.stem)
        tiles = [labeled_tile(image, f"Original - ImageID {image_id}", args.tile_size)]
        for name, masks in submissions:
            mask = rle_decode(masks[image_id], image.shape[:2])
            if args.mode == "mask":
                view = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
            else:
                view = image.copy()
                view[mask] = (image[mask] * 0.55 + np.array([0, 255, 0]) * 0.45).astype(np.uint8)
            tiles.append(labeled_tile(view, name, args.tile_size, mask=args.mode == "mask"))
        blank = np.full_like(tiles[0], 255)
        tiles.extend([blank] * (rows * columns - len(tiles)))
        grid = np.concatenate([
            np.concatenate(tiles[row * columns:(row + 1) * columns], axis=1)
            for row in range(rows)
        ], axis=0)
        output = args.output_dir / f"{image_id}.jpg"
        if not cv2.imwrite(str(output), grid, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"Failed to save {output}")
    print(f"Saved {len(images)} comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
