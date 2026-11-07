import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def experiment_name(source: Path) -> str:
    metadata = source.with_suffix(source.suffix + ".json")
    if metadata.is_file():
        return json.loads(metadata.read_text())["experiment"]
    for directory in source.parents:
        config = directory / ".hydra/config.yaml"
        if config.is_file():
            return str(OmegaConf.load(config).name)
    return source.parent.name


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = experiment_name(args.checkpoint)
    output = args.output or Path("predictions") / "test" / name / "submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint).to(device).eval()
    dataloader = DataLoader(TestDataset(args.test_dir), batch_size=1, num_workers=4)
    write_submission(
        model, dataloader, output, device,
        threshold=args.threshold, hflip_tta=args.hflip_tta,
        input_size=tuple(args.input_size) if args.input_size else None,
    )
    output.with_suffix(output.suffix + ".json").write_text(json.dumps({
        "experiment": name, "checkpoint": str(args.checkpoint),
        "input_size": args.input_size, "threshold": args.threshold, "hflip_tta": args.hflip_tta,
    }, indent=2) + "\n")
    print(f"Saved {output}")
    if args.visualize:
        from src.visualize import visualize_submission

        visualize_submission(argparse.Namespace(
            submission=output, test_dir=Path(args.test_dir),
            output_dir=output.parent / output.stem,
            num_images=args.num_images, seed=args.seed,
        ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--test-dir", default="data/test/test/images")
    parser.add_argument("--output", type=Path, help="Default: predictions/test/<experiment>/submission.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hflip-tta", action="store_true")
    parser.add_argument("--input-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--visualize", action="store_true", help="Save preview PNGs after creating submission")
    parser.add_argument("--num-images", type=int, default=20, help="Number of preview images with --visualize")
    parser.add_argument("--seed", type=int, default=42, help="Seed for preview image selection")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if args.input_size and any(size <= 0 for size in args.input_size):
        parser.error("--input-size dimensions must be positive")
    if args.num_images < 1:
        parser.error("--num-images must be positive")
    return args


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    run_dir = checkpoint_path.parent.parent
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if model_cfg.get("_target_") == "src.models.SMPModel":
        # A checkpoint already includes encoder weights; inference needs no download.
        model_cfg.encoder_weights = None
    model = instantiate(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict({
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    })
    return model


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module, images: torch.Tensor,
    hflip_tta: bool = False, input_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    output_size = images.shape[-2:]
    if input_size is not None and tuple(input_size) != tuple(output_size):
        images = F.interpolate(images, size=input_size, mode="bilinear", align_corners=False)
    probs = model(images).sigmoid()
    if hflip_tta:
        flipped_probs = model(images.flip(-1)).sigmoid().flip(-1)
        probs = (probs + flipped_probs) / 2
    if probs.shape[-2:] != output_size:
        probs = F.interpolate(probs, size=output_size, mode="bilinear", align_corners=False)
    return probs[:, 0]


@torch.no_grad()
def write_submission(
    model: torch.nn.Module, dataloader: DataLoader, output: str, device: str,
    threshold: float = 0.5, hflip_tta: bool = False,
    input_size: tuple[int, int] | None = None,
) -> None:
    with open(output, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["ImageID", "RLE"])
        for image, image_id in tqdm(dataloader):
            probs = predict_probabilities(model, image.to(device), hflip_tta, input_size)
            mask = probs[0] > threshold
            writer.writerow([image_id.item(), rle_encode(mask.cpu().numpy())])


def rle_encode(mask: np.ndarray) -> str:
    flat = np.ravel(mask, order="F").astype(np.uint8)
    if flat.sum() == 0:
        return "EMPTY"

    flat = np.concatenate(([0], flat, [0]))
    changes = np.where(flat[1:] != flat[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes.tolist()))


class TestDataset(Dataset):
    def __init__(self, root: str) -> None:
        self.root = root

    def __len__(self) -> int:
        return len(os.listdir(self.root))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = cv2.imread(os.path.join(self.root, f"{idx}.png"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1).float(), idx


if __name__ == "__main__":
    main()
