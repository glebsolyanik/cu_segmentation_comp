from typing import Callable

import lightning as L
import torch
import torch.nn as nn
import torchmetrics.segmentation as tms


class LightningPipeline(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: Callable,
        scheduler: Callable | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self._optimizer = optimizer
        self._scheduler = scheduler

        self.train_metric = tms.MeanIoU(num_classes=2, input_format="index")
        self.valid_metric = tms.MeanIoU(num_classes=2, input_format="index")

    def configure_optimizers(self) -> dict:
        optimizer = self._optimizer(self.parameters())
        if self._scheduler is None:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": self._scheduler(optimizer), "interval": "epoch", "frequency": 1},
        }

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, "valid")

    def shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        images, masks = batch
        mask_logits = self(images)
        loss = self.loss_fn(mask_logits, masks)

        pred_masks = (mask_logits.sigmoid() > 0.5).long()
        metric = getattr(self, f"{stage}_metric")
        metric(pred_masks, masks.long())

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(f"{stage}_iou", metric, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images).squeeze(1)
