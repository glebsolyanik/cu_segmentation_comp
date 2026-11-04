import tempfile
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.data import PetDataModule, PetDataset, build_transform
from src.losses import BCEDiceLoss
from src.models import BasicUnet
from src.predict import load_model, predict_probabilities


ROOT = Path(__file__).resolve().parents[1]


def config(experiment):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=[f"experiment={experiment}"])


class ExperimentTests(unittest.TestCase):
    def test_configs_change_one_factor(self):
        baseline = config("002_augmentations")
        for name in ("003_bce_dice", "004_cosine", "007_resolution384"):
            candidate = config(name)
            self.assertEqual(candidate.model, baseline.model)
            self.assertEqual(candidate.optimizer, baseline.optimizer)
            self.assertEqual(candidate.data.train_augmentations, baseline.data.train_augmentations)
        scratch = config("005_unet_resnet18_scratch")
        pretrained = config("006_unet_resnet18_pretrained")
        pretrained.model.encoder_weights = None
        self.assertEqual(scratch.model, pretrained.model)

    def test_dice_loss_extremes_and_gradients(self):
        criterion = BCEDiceLoss()
        for target in (0.0, 1.0):
            for value in (-100.0, 0.0, 100.0):
                logits = torch.full((2, 8, 8), value, requires_grad=True)
                loss = criterion(logits, torch.full_like(logits, target))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(logits.grad).all())
        target = torch.ones(2, 8, 8)
        self.assertLess(criterion(torch.full_like(target, 10), target),
                        criterion(torch.full_like(target, -10), target))

    def test_cosine_reaches_configured_minimum(self):
        cfg = config("004_cosine")
        optimizer = instantiate(cfg.optimizer)([torch.nn.Parameter(torch.zeros(1))])
        scheduler = instantiate(cfg.scheduler)(optimizer)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        for _ in range(cfg.trainer.max_epochs):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)

    def test_tta_unflips_prediction_and_restores_size(self):
        class PointwiseModel(torch.nn.Module):
            def forward(self, images):
                return images[:, :1] * 2 - 1

        images = torch.rand(2, 3, 32, 48)
        model = PointwiseModel()
        plain = predict_probabilities(model, images)
        tta = predict_probabilities(model, images, hflip_tta=True)
        torch.testing.assert_close(tta, plain)
        resized = predict_probabilities(model, images, hflip_tta=True, input_size=(64, 64))
        self.assertEqual(resized.shape, (2, 32, 48))
        self.assertTrue(((resized >= 0) & (resized <= 1)).all())

    def test_old_checkpoint_still_loads(self):
        model = BasicUnet(base_channels=2).eval()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / ".hydra").mkdir()
            (run / "checkpoints").mkdir()
            OmegaConf.save(OmegaConf.create({"model": {
                "_target_": "src.models.BasicUnet", "base_channels": 2,
            }}), run / ".hydra/config.yaml")
            checkpoint = run / "checkpoints/model.ckpt"
            torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, checkpoint)
            restored = load_model(checkpoint).eval()
            image = torch.rand(1, 3, 32, 32)
            with torch.no_grad():
                torch.testing.assert_close(restored(image), model(image))

    @unittest.skipUnless((ROOT / "data/train/data").exists(), "Local Pet dataset not available")
    def test_resolution_uses_identical_validation_targets(self):
        targets = []
        for experiment in ("002_augmentations", "007_resolution384"):
            cfg = config(experiment)
            cfg.data.root = str(ROOT / "data/train/data")
            dm = PetDataModule(**cfg.data, valid_mask_size=cfg.evaluation.image_size)
            dm.setup("fit")
            image, mask = dm.valid_dataset[0]
            self.assertEqual(tuple(image.shape[-2:]), tuple(cfg.data.image_size))
            self.assertEqual(tuple(mask.shape), (256, 256))
            targets.append(mask)
        self.assertTrue(torch.equal(*targets))
        legacy = PetDataset(str(ROOT / "data/train/data"), "test", build_transform((256, 256)))
        self.assertTrue(torch.equal(targets[0], legacy[0][1]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
