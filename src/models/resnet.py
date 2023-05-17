import torch.nn.functional as F  # noqa: N812
import torch.optim
import torchvision
from torch import nn

from .modules import ImageClassifier


class Resnet(ImageClassifier):
    def __init__(
        self,
        image_size=None,
        num_channels=3,
        num_classes=10,
    ):
        super().__init__(
            image_size=image_size,
            num_channels=num_channels,
            num_classes=num_classes,
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

    def configure_optimizers(self):
        return self.create_optimizers(
            optimizer_cls=torch.optim.SGD,
            optimizer_hparams={
                "lr": 0.05,
                "momentum": 0.9,
                "weight_decay": 5e-4,
            },
            scheduler_cls=torch.optim.lr_scheduler.OneCycleLR,
            scheduler_hparams={"max_lr": 0.1},
            scheduler_interval="step",
            add_total_steps=True,
        )
