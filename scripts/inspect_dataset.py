"""Create contact sheets of originals and ground-truth overlays for manual inspection."""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/train/data"))
    parser.add_argument("--split", choices=("trainval", "test"), default="trainval")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--names", nargs="+", help="Inspect specific image names at full size")
    args = parser.parse_args()
    annotations = args.data_root / "annotations/annotations"
    names = args.names or [line.split()[0] for line in (annotations / f"{args.split}.txt").read_text().splitlines() if line.strip()]
    output = args.output_dir or Path("predictions/dataset") / args.split
    output.mkdir(parents=True, exist_ok=True)
    size, per_page, columns = (512, 1, 1) if args.names else (192, 48, 6)
    index = []
    for offset in tqdm(range(0, len(names), per_page), desc="Dataset sheets"):
        group = names[offset:offset + per_page]
        page = offset // per_page + 1
        sheet = np.full((math.ceil(len(group) / columns) * (size + 28), columns * size * 2, 3), 255, np.uint8)
        for position, name in enumerate(group):
            image_path = args.data_root / "images/images" / f"{name}.jpg"
            mask_path = annotations / "trimaps" / f"{name}.png"
            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise OSError(f"Cannot read {image_path} or {mask_path}")
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            foreground = (mask == 1) | (mask == 3)
            overlay = image.copy()
            overlay[foreground] = (image[foreground] * 0.55 + np.array([0, 255, 0]) * 0.45).astype(np.uint8)
            row, col = divmod(position, columns)
            y, x = row * (size + 28), col * size * 2
            cv2.putText(sheet, name, (x + 4, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            sheet[y + 28:y + 28 + size, x:x + size] = image
            sheet[y + 28:y + 28 + size, x + size:x + 2 * size] = overlay
            index.append((f"page_{page:03d}.jpg", name, str(image_path), str(mask_path)))
        path = output / f"page_{page:03d}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93]):
            raise OSError(f"Cannot save {path}")
    with (output / "index.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["page", "image_name", "image_path", "mask_path"])
        writer.writerows(index)
    print(f"Saved {len(names)} images to {output}; left = original, right = ground truth")


if __name__ == "__main__":
    main()
