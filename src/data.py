import os
from typing import Callable

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PetDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        image_size: tuple[int, int],
        batch_size: int,
        train_workers: int,
        valid_workers: int,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.valid_workers = valid_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str) -> None:
        self.train_dataset = PetDataset(self.root, "trainval", build_transform(self.image_size))
        self.valid_dataset = PetDataset(self.root, "test", build_transform(self.image_size))

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


def build_transform(image_size: tuple[int, int]) -> A.Compose:
    return A.Compose([
        A.Resize(*image_size),
        A.ToFloat(),
        A.ToTensorV2(transpose_mask=True),
    ])


class PetDataset(Dataset):
    BORDER = 1
    PET = 3

    def __init__(self, root: str, split: str, transform: Callable | None = None) -> None:
        super().__init__()
        self.transform = transform
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

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"].float()
        return image, mask

    def _load_image(self, name: str) -> np.ndarray:
        image = cv2.imread(os.path.join(self.images_dir, f"{name}.jpg"))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, name: str) -> np.ndarray:
        trimap = cv2.imread(os.path.join(self.masks_dir, f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        return ((trimap == self.PET) | (trimap == self.BORDER)).astype(np.uint8)
