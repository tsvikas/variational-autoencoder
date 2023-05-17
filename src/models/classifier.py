import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from torchmetrics.functional.classification import multiclass_accuracy


class NLLClassifier(pl.LightningModule):
    def __init__(
        self,
        image_size,
        num_channels,
        num_classes,
    ):
        super().__init__()
        sample_batch_size = 32
        self.image_size = image_size or 96
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.example_input_array = torch.empty(
            sample_batch_size, num_channels, self.image_size, self.image_size
        )

    def step(self, batch, stage, *, evaluate=False):
        x, target = batch
        nll = self(x)
        loss = F.nll_loss(nll, target)
        self.log(f"{stage}_loss", loss, prog_bar=evaluate)
        if evaluate:
            acc = multiclass_accuracy(nll, target, num_classes=self.num_classes)
            self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", evaluate=True)

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", evaluate=True)


class NLLClassifierWithOptimizer(NLLClassifier):
    def create_optimizers(
        self,
        optimizer_cls,
        optimizer_hparams=None,
        scheduler_cls=None,
        scheduler_interval="epoch",
        scheduler_hparams=None,
        add_total_steps=False,
    ):
        # optimizer
        optimizer_hparams = optimizer_hparams or {}
        optimizer = optimizer_cls(self.parameters(), **optimizer_hparams)
        self.logger.log_hyperparams(
            {"optimizer": optimizer_cls.__name__, **optimizer_hparams}
        )
        if not scheduler_cls:
            return {"optimizer": optimizer}

        # scheduler
        if scheduler_interval not in ["epoch", "step"]:
            raise ValueError("scheduler_interval not in ['epoch', 'step']")
        scheduler_hparams = scheduler_hparams or {}
        if add_total_steps:
            scheduler_hparams["total_steps"] = self.trainer.estimated_stepping_batches
        lr_scheduler = {
            "scheduler": scheduler_cls(optimizer, **scheduler_hparams),
            "interval": scheduler_interval,
        }
        self.logger.log_hyperparams(
            {
                "scheduler": scheduler_cls.__name__,
                "scheduler_interval": scheduler_interval,
                **scheduler_hparams,
            }
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}