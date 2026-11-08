import os
import json
from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class PetDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        image_size: tuple[int, int],
        batch_size: int,
        train_workers: int,
        valid_workers: int,
        pin_memory: bool,
        train_augmentations: list | None = None,
        valid_mask_size: tuple[int, int] | None = None,
        external_root: str | None = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.valid_workers = valid_workers
        self.pin_memory = pin_memory
        self.train_augmentations = [instantiate(transform) for transform in (train_augmentations or [])]
        self.valid_mask_size = valid_mask_size
        self.external_root = external_root

    def setup(self, stage: str) -> None:
        self.train_dataset = PetDataset(
            self.root, "trainval", build_transform(self.image_size, self.train_augmentations)
        )
        if self.external_root:
            external = ExternalPetDataset(
                self.external_root, build_transform(self.image_size, self.train_augmentations)
            )
            self.train_dataset = ConcatDataset([self.train_dataset, external])
        self.valid_dataset = PetDataset(
            self.root, "test", build_transform(self.image_size), mask_size=self.valid_mask_size
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, self.train_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.valid_dataset, self.valid_workers, shuffle=False)

    def _dataloader(self, dataset: Dataset, num_workers: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=num_workers > 0,
        )


def build_transform(
    image_size: tuple[int, int], augmentations: list[A.BasicTransform] | None = None
) -> A.Compose:
    return A.Compose([
        A.Resize(*image_size),
        *(augmentations or []),
        A.ToFloat(),
        A.ToTensorV2(transpose_mask=True),
    ])


class ExternalPetDataset(Dataset):
    """Prepared external binary masks: 1 = cat/dog, 0 = everything else."""

    def __init__(self, root: str, transform: Callable) -> None:
        self.root = Path(root)
        self.transform = transform
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("training_allowed") is False:
            raise ValueError(f"External dataset is marked as unsuitable for training: {root}")
        self.samples = manifest["samples"]
        if not self.samples:
            raise ValueError(f"No external training examples in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = cv2.imread(str(self.root / sample["image"]))
        mask = cv2.imread(str(self.root / sample["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise OSError(f"Cannot read external sample {sample['name']}")
        if image.shape[:2] != mask.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Invalid external binary mask: {sample['name']}")
        result = self.transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask=mask)
        return result["image"], result["mask"].float()


class PetDataset(Dataset):
    BORDER = 1
    PET = 3

    def __init__(
        self, root: str, split: str, transform: Callable | None = None,
        mask_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.mask_size = mask_size
        self.images_dir = os.path.join(root, "images", "images")
        annotations_dir = os.path.join(root, "annotations", "annotations")
        self.masks_dir = os.path.join(annotations_dir, "trimaps")

        with open(os.path.join(annotations_dir, f"{split}.txt")) as file:
            self.image_names = [line.split(" ")[0] for line in file]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.image_names[idx]
        image = self._load_image(name)
        mask = self._load_mask(name)
        # Resize scoring targets directly from the original annotation, never
        # through the model's input resolution (e.g. original -> 384 -> 256).
        score_mask = None
        if self.mask_size is not None:
            height, width = self.mask_size
            score_mask = torch.from_numpy(
                cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            ).float()

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"].float()
        if score_mask is not None:
            mask = score_mask
        return image, mask

    def _load_image(self, name: str) -> np.ndarray:
        image = cv2.imread(os.path.join(self.images_dir, f"{name}.jpg"))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, name: str) -> np.ndarray:
        trimap = cv2.imread(os.path.join(self.masks_dir, f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        return ((trimap == self.PET) | (trimap == self.BORDER)).astype(np.uint8)
