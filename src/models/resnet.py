import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import nn

from . import base


class Resnet(base.ImageClassifier):
    def __init__(
        self,
        image_size=None,
        num_channels=3,
        num_classes=10,
        *,
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        scheduler_interval="epoch",
        scheduler_add_total_steps=False,
        scheduler_monitor=None,
    ):
        super().__init__(
            image_size=image_size,
            num_channels=num_channels,
            num_classes=num_classes,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_interval=scheduler_interval,
            scheduler_add_total_steps=scheduler_add_total_steps,
            scheduler_monitor=scheduler_monitor,
        )
        self.save_hyperparameters()
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
