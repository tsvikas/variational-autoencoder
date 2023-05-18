import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from torchmetrics.functional.classification import multiclass_accuracy


class SimpleLightningModule(pl.LightningModule):
    @classmethod
    def load_latest_checkpoint(cls, base_dir, **kwargs):
        all_checkpoints = sorted(
            base_dir.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime
        )
        if not all_checkpoints:
            raise RuntimeError("no checkpoints found")
        ckpt_fn_latest = all_checkpoints[-1]
        return cls.load_from_checkpoint(ckpt_fn_latest, **kwargs)

    def step(self, batch, batch_idx, stage, *, evaluate=False):
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
        scheduler_interval_options = ["epoch", "step"]
        if scheduler_interval not in scheduler_interval_options:
            raise ValueError(f"scheduler_interval not in {scheduler_interval_options}")
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


class NLLClassifier(SimpleLightningModule):
    """
    classifier that returns the negative-log-likelihood of each class.
    uses nll_loss
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def step(self, batch, batch_idx, stage, *, evaluate=False):
        x, target = batch
        nll = self(x)
        loss = F.nll_loss(nll, target)
        self.log(f"loss/{stage}", loss, prog_bar=evaluate)
        if evaluate:
            acc = multiclass_accuracy(nll, target, num_classes=self.num_classes)
            self.log(f"accuracy/{stage}", acc, prog_bar=True)
        return loss


class ImageClassifier(NLLClassifier):
    def __init__(self, image_size, num_channels, num_classes):
        super().__init__(num_classes)
        sample_batch_size = 32
        self.image_size = image_size or 96
        self.num_channels = num_channels
        self.example_input_array = torch.empty(
            sample_batch_size, num_channels, self.image_size, self.image_size
        )


class AutoEncoder(SimpleLightningModule):
    """
    encoder that tries to return itself.
    uses mse_loss
    """

    n_images_to_save = 4

    def step(self, batch, batch_idx, stage, *, evaluate=False):
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
    def __init__(self, image_size, num_channels):
        super().__init__()
        sample_batch_size = 32
        self.image_size = image_size or 96
        self.num_channels = num_channels
        self.example_input_array = torch.empty(
            sample_batch_size, num_channels, self.image_size, self.image_size
        )
