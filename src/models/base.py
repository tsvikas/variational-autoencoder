from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from torchmetrics.functional.classification import multiclass_accuracy


class SimpleLightningModule(pl.LightningModule):
    @classmethod
    def load_latest_checkpoint(cls, base_dir: Path, **kwargs):
        all_checkpoints = sorted(
            base_dir.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime
        )
        if not all_checkpoints:
            raise RuntimeError("no checkpoints found")
        ckpt_fn_latest = all_checkpoints[-1]
        return cls.load_from_checkpoint(ckpt_fn_latest, **kwargs)

    def step(self, batch, batch_idx, stage: str, *, evaluate=False):
        """
        run a general training/validation/test step.
        return the loss
        evaluate is True if stage is validation or test
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "training")

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, "validation", evaluate=True)

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx, "test", evaluate=True)


class LightningModuleWithOptimizer(SimpleLightningModule):
    def __init__(self, optimizer_cls=None, optimizer_kwargs=None):
        super().__init__()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}

    def configure_optimizers(self):
        if self.optimizer_cls is None:
            raise ValueError("must define an optimizer_cls")
        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        self.logger.log_hyperparams({"optimizer": self.optimizer_cls.__name__})
        return optimizer


class LightningModuleWithScheduler(LightningModuleWithOptimizer):
    def __init__(
        self,
        *,
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        scheduler_interval="epoch",
        scheduler_add_total_steps=False,
    ):
        super().__init__(optimizer_cls=optimizer_cls, optimizer_kwargs=optimizer_kwargs)
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler_interval = scheduler_interval
        self.scheduler_add_total_steps = scheduler_add_total_steps

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()

        scheduler_interval_options = ["epoch", "step"]
        if self.scheduler_interval not in scheduler_interval_options:
            raise ValueError(f"scheduler_interval not in {scheduler_interval_options}")
        created_scheduler_kwargs = {}
        if self.scheduler_add_total_steps:
            total_steps = self.trainer.estimated_stepping_batches
            created_scheduler_kwargs["total_steps"] = total_steps
        lr_scheduler = {
            "scheduler": self.scheduler_cls(
                optimizer, **self.scheduler_kwargs, **created_scheduler_kwargs
            ),
            "interval": self.scheduler_interval,
        }
        self.logger.log_hyperparams(
            {
                "scheduler": self.scheduler_cls.__name__,
                "scheduler_interval": self.scheduler_interval,
            }
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class NLLClassifier(LightningModuleWithScheduler):
    """
    classifier that returns the negative-log-likelihood of each class.
    uses nll_loss
    """

    def __init__(
        self,
        num_classes: int,
        *,
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        scheduler_interval="epoch",
        scheduler_add_total_steps=False,
    ):
        super().__init__(
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_interval=scheduler_interval,
            scheduler_add_total_steps=scheduler_add_total_steps,
        )
        self.num_classes = num_classes

    def step(self, batch, batch_idx, stage: str, *, evaluate=False):
        x, target = batch
        nll = self(x)
        loss = F.nll_loss(nll, target)
        self.log(f"loss/{stage}", loss, prog_bar=evaluate)
        if evaluate:
            acc = multiclass_accuracy(nll, target, num_classes=self.num_classes)
            self.log(f"accuracy/{stage}", acc, prog_bar=True)
        return loss


class ImageClassifier(NLLClassifier):
    def __init__(
        self,
        image_size: int,
        num_channels: int,
        num_classes: int,
        *,
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        scheduler_interval="epoch",
        scheduler_add_total_steps=False,
    ):
        super().__init__(
            num_classes=num_classes,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_interval=scheduler_interval,
            scheduler_add_total_steps=scheduler_add_total_steps,
        )
        sample_batch_size = 32
        self.image_size = image_size or 96
        self.num_channels = num_channels
        self.example_input_array = torch.empty(
            sample_batch_size, num_channels, self.image_size, self.image_size
        )


class AutoEncoder(LightningModuleWithScheduler):
    """
    encoder that tries to return itself.
    uses mse_loss
    """

    n_images_to_save = 4

    def step(self, batch, batch_idx, stage: str, *, evaluate=False):
        x, target = batch
        x2 = self(x)
        loss = F.mse_loss(x2, x)
        self.log(f"loss/{stage}", loss, prog_bar=evaluate)
        if stage == "validation":
            if self.global_step == 0 and batch_idx == 0:
                self.logger.log_image("image/src", list(x[: self.n_images_to_save]))
            if batch_idx == 0:
                self.logger.log_image("image/pred", list(x2[: self.n_images_to_save]))
        return loss


class ImageAutoEncoder(AutoEncoder):
    def __init__(
        self,
        image_size: int,
        num_channels: int,
        *,
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        scheduler_interval="epoch",
        scheduler_add_total_steps=False,
    ):
        super().__init__(
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_interval=scheduler_interval,
            scheduler_add_total_steps=scheduler_add_total_steps,
        )
        sample_batch_size = 32
        self.image_size = image_size or 96
        self.num_channels = num_channels
        self.example_input_array = torch.empty(
            sample_batch_size, num_channels, self.image_size, self.image_size
        )
