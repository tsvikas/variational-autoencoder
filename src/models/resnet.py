import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import nn, optim

from .classifier import NLLClassifier


class Resnet(NLLClassifier):
    def __init__(
        self,
        num_channels=3,
        num_classes=10,
        *,
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4,
        max_lr=0.1,
    ):
        super().__init__(num_channels=num_channels, num_classes=num_classes)
        self.save_hyperparameters(ignore=["num_classes", "num_channels"])
        self.optimizer_hparams = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
        self.scheduler_hparams = {"max_lr": max_lr}
        # create the model
        self.model = torchvision.models.resnet18(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        batch_size, channels, _height, _width = x.shape
        assert channels == self.num_channels
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        assert x.shape == (batch_size, self.num_classes)
        return x

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.optimizer_hparams)
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                total_steps=self.trainer.estimated_stepping_batches,
                **self.scheduler_hparams,
            ),
            "interval": "step",
        }
        # self.save_hyperparameters({"optimizer": type(optimizer).__name__})
        # self.save_hyperparameters(
        #     {"scheduler": type(lr_scheduler["scheduler"]).__name__}
        # )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
