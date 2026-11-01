import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.checkpoint).to(device).eval()
    dataloader = DataLoader(TestDataset(args.test_dir), batch_size=1, num_workers=4)
    write_submission(model, dataloader, args.output, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--test-dir", default="data/test/test/images")
    parser.add_argument("--output", default="submission.csv")
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    run_dir = checkpoint_path.parent.parent
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    model = instantiate(cfg.model)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict({
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    })
    return model


@torch.no_grad()
def write_submission(model: torch.nn.Module, dataloader: DataLoader, output: str, device: str) -> None:
    with open(output, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["ImageID", "RLE"])
        for image, image_id in tqdm(dataloader):
            mask = model(image.to(device))[0, 0].sigmoid() > 0.5
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
