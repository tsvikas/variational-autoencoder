import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from torchmetrics.functional.classification import multiclass_accuracy


class NLLClassifier(pl.LightningModule):
    def __init__(
        self,
        num_channels=3,
        num_classes=10,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.example_input_array = torch.empty(32, num_channels, 28, 28)

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
