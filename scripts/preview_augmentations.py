"""Preview the actual training augmentation config on images and ground-truth masks."""

import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def tile(image, mask, title):
    overlay = image.copy()
    selected = mask.astype(bool)
    overlay[selected] = (image[selected] * 0.55 + np.array([0, 255, 0]) * 0.45).astype(np.uint8)
    panel = cv2.copyMakeBorder(overlay, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.num_images < 1:
        parser.error("num-images must be positive")
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        cfg = compose(config_name="config", overrides=[f"experiment={args.experiment}"])
    height, width = cfg.data.image_size
    transforms = A.Compose([A.Resize(height, width), *[instantiate(item) for item in cfg.data.train_augmentations]])
    transforms.set_random_seed(args.seed)
    resize = A.Resize(height, width)
    data = Path(cfg.data.root)
    annotations = data / "annotations/annotations"
    names = [line.split()[0] for line in (annotations / "trainval.txt").read_text().splitlines() if line.strip()]
    output = args.output_dir or Path("predictions/augmentations") / cfg.name
    output.mkdir(parents=True, exist_ok=True)
    for index in np.random.default_rng(args.seed).permutation(len(names))[:args.num_images]:
        name = names[int(index)]
        image = cv2.imread(str(data / "images/images" / f"{name}.jpg"))
        trimap = cv2.imread(str(annotations / "trimaps" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or trimap is None:
            raise OSError(f"Cannot read image/mask: {name}")
        mask = ((trimap == 1) | (trimap == 3)).astype(np.uint8)
        original = resize(image=image, mask=mask)
        panels = [tile(original["image"], original["mask"], "Original + ground truth")]
        for variant in range(5):
            result = transforms(image=image, mask=mask)
            if not set(np.unique(result["mask"])).issubset({0, 1}):
                raise ValueError("Augmentation corrupted binary mask labels")
            panels.append(tile(result["image"], result["mask"], f"Augmentation {variant + 1}"))
        grid = np.concatenate([np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)], axis=0)
        path = output / f"{name}.jpg"
        if not cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 93]):
            raise OSError(f"Cannot save {path}")
        print(path)


if __name__ == "__main__":
    main()
