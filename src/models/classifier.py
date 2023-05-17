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


def parse_name_or_cls(name_or_cls, namespace):
    if isinstance(name_or_cls, str):
        name = name_or_cls
        cls = getattr(namespace, name)
    else:
        cls = name_or_cls
        name = cls.__name__
    return name, cls


class NLLClassifierWithOptimizer(NLLClassifier):
    def __init__(
        self,
        image_size,
        num_channels,
        num_classes,
        optimizer_name_or_cls,
        optimizer_hparams=None,
        scheduler_name_or_cls=None,
        scheduler_hparams=None,
        scheduler_interval="epoch",
        add_total_steps=False,
    ):
        super().__init__(
            image_size=image_size, num_channels=num_channels, num_classes=num_classes
        )
        # optimizer
        self.optimizer_name, self.optimizer_cls = parse_name_or_cls(
            optimizer_name_or_cls, torch.optim
        )
        self.optimizer_hparams = optimizer_hparams or {}
        self.save_hyperparameters(
            {"optimizer": self.optimizer_name, **optimizer_hparams}
        )
        # scheduler
        self.use_scheduler = scheduler_name_or_cls is not None
        if self.use_scheduler:
            self.scheduler_name, self.scheduler_cls = parse_name_or_cls(
                scheduler_name_or_cls, torch.optim.lr_scheduler
            )
            self.scheduler_hparams = scheduler_hparams or {}
            self.scheduler_interval = scheduler_interval
            self.add_total_steps = add_total_steps
            self.save_hyperparameters(
                {
                    "scheduler": self.scheduler_name,
                    "scheduler_interval": scheduler_interval,
                    **scheduler_hparams,
                }
            )

    def configure_optimizers(self):
        # optimizer
        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_hparams)
        if not self.use_scheduler:
            return {"optimizer": optimizer}
        # scheduler
        scheduler_hparams = self.scheduler_hparams
        if self.add_total_steps:
            scheduler_hparams["total_steps"] = self.trainer.estimated_stepping_batches
        lr_scheduler = {
            "scheduler": self.scheduler_cls(optimizer, **scheduler_hparams),
            "interval": self.scheduler_interval,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
