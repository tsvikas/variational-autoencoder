import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from pytorch_lightning import LightningModule
from torch import optim
from torchmetrics.functional.classification import multiclass_accuracy


def create_model(num_classes, in_channels=3):
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model


class Resnet(LightningModule):
    def __init__(self, num_classes=10, lr=0.05):
        super().__init__()
        self.save_hyperparameters("lr")
        self.lr = lr
        self.num_classes = num_classes
        self.model = create_model(num_classes=num_classes)
        self.example_input_array = torch.empty(1, 3, 224, 224)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        predictions = torch.argmax(logits, dim=1)
        acc = multiclass_accuracy(predictions, y, num_classes=self.num_classes)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler_dict = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.1,
                total_steps=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
