import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import nn

from .classifier import NLLClassifierWithOptimizer


class Resnet(NLLClassifierWithOptimizer):
    def __init__(
        self,
        image_size=None,
        num_channels=3,
        num_classes=10,
        *,
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4,
        max_lr=0.1,
    ):
        optimizer_hparams = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
        scheduler_hparams = {"max_lr": max_lr}
        super().__init__(
            image_size=image_size,
            num_channels=num_channels,
            num_classes=num_classes,
            optimizer_name_or_cls="SGD",
            optimizer_hparams=optimizer_hparams,
            scheduler_name_or_cls="OneCycleLR",
            scheduler_hparams=scheduler_hparams,
            scheduler_interval="step",
            add_total_steps=True,
        )
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
