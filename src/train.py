from pathlib import Path

import hydra
import lightning as L
from dotenv import load_dotenv
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.data import PetDataModule
from src.pipeline import LightningPipeline


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed)

    datamodule = PetDataModule(**cfg.data)
    pipeline = LightningPipeline(
        model=instantiate(cfg.model),
        loss=instantiate(cfg.loss),
        optimizer=instantiate(cfg.optimizer),
        scheduler=instantiate(cfg.scheduler) if cfg.scheduler else None,
    )
    logger = build_logger(cfg)
    trainer = L.Trainer(**cfg.trainer, logger=logger, callbacks=build_callbacks(cfg, logger))

    trainer.fit(pipeline, datamodule=datamodule)


def build_logger(cfg: DictConfig):
    if cfg.logger is None:
        return False

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    logger = instantiate(cfg.logger)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    return logger


def build_callbacks(cfg: DictConfig, logger) -> list:
    callbacks = [
        ModelCheckpoint(dirpath=cfg.checkpoint_dir, monitor="valid_iou", mode="max", save_top_k=1, save_last=True),
    ]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return callbacks


if __name__ == "__main__":
    main()
