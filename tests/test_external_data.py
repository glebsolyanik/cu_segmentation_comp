import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.data import ExternalPetDataset, PetDataModule
from scripts.prepare_coco_people import candidates


class ExternalDataTests(unittest.TestCase):
    def test_rejected_external_data_cannot_be_used_for_training(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "manifest.json").write_text(json.dumps({
                "training_allowed": False, "samples": [{"name": "test_overlap"}],
            }))
            with self.assertRaisesRegex(ValueError, "unsuitable for training"):
                ExternalPetDataset(directory, transform=lambda **kwargs: kwargs)

    def test_crowd_pets_are_not_mislabeled_as_negatives(self):
        def ann(image, category, crowd=0):
            return {"image_id": image, "category_id": category, "iscrowd": crowd,
                    "segmentation": [[1, 1, 3, 1, 3, 3]]}
        data = {"categories": [{"id": 1, "name": "person"}, {"id": 17, "name": "cat"}, {"id": 18, "name": "dog"}],
                "images": [{"id": i} for i in range(1, 5)],
                "annotations": [ann(1, 1), ann(1, 18), ann(2, 1),
                                ann(3, 1), ann(3, 18, crowd=1), ann(4, 18)]}
        groups = candidates(data)
        self.assertEqual([image["id"] for image, _ in groups["person_pet"]], [1])
        self.assertEqual([image["id"] for image, _ in groups["person_only"]], [2])

    def test_external_masks_and_validation_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oxford, external = root / "oxford", root / "external"
            images = oxford / "images/images"
            annotations = oxford / "annotations/annotations"
            images.mkdir(parents=True)
            (annotations / "trimaps").mkdir(parents=True)
            image = np.full((32, 32, 3), 120, np.uint8)
            cv2.imwrite(str(images / "pet.jpg"), image)
            trimap = np.full((32, 32), 2, np.uint8)
            trimap[8:24, 8:24] = 1
            cv2.imwrite(str(annotations / "trimaps/pet.png"), trimap)
            for split in ("trainval", "test"):
                (annotations / f"{split}.txt").write_text("pet 1 1 1\n")
            external.mkdir()
            samples = []
            for name in ("positive", "negative"):
                mask = np.zeros((32, 32), np.uint8)
                if name == "positive":
                    mask[8:24, 8:24] = 1
                cv2.imwrite(str(external / f"{name}.jpg"), image)
                cv2.imwrite(str(external / f"{name}.png"), mask)
                samples.append({"name": name, "image": f"{name}.jpg", "mask": f"{name}.png"})
            (external / "manifest.json").write_text(json.dumps({"samples": samples}))
            options = dict(root=str(oxford), image_size=(32, 32), batch_size=1,
                           train_workers=0, valid_workers=0, pin_memory=False)
            plain = PetDataModule(**options)
            mixed = PetDataModule(**options, external_root=str(external))
            plain.setup("fit")
            mixed.setup("fit")
            self.assertEqual(len(mixed.train_dataset), 3)
            self.assertEqual(len(mixed.valid_dataset), len(plain.valid_dataset))
            torch.testing.assert_close(mixed.valid_dataset[0][1], plain.valid_dataset[0][1])
            self.assertGreater(mixed.train_dataset[1][1].sum(), 0)
            self.assertEqual(mixed.train_dataset[2][1].sum(), 0)


if __name__ == "__main__":
    unittest.main()
