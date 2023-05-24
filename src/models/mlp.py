# https://github.com/rubentea16/pl-mnist/blob/master/model.py
# https://docs.ray.io/en/latest/tune/examples/includes/mnist_ptl_mini.html
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from . import base


class MultiLayerPerceptron(base.ImageClassifier):
    def __init__(
        self,
        image_size=28,
        num_channels=3,
        num_classes=10,
        *,
        hidden_size_1=128,
        hidden_size_2=256,
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        scheduler_interval="epoch",
        scheduler_add_total_steps=False,
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
        )
        self.save_hyperparameters()
        # create the model
        self.layer_1 = nn.Linear(num_channels * image_size * image_size, hidden_size_1)
        self.layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer_3 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert channels == self.num_channels
        assert height == width == self.image_size
        # (b, c, 28, 28) -> (b, c*28*28)
        x = x.view(batch_size, -1)
        # -> (b, 128)
        x = self.layer_1(x)
        x = F.relu(x)
        # -> (b, 256)
        x = self.layer_2(x)
        x = F.relu(x)
        # -> (b, 10)
        x = self.layer_3(x)
        # log-probability distribution over labels
        x = F.log_softmax(x, dim=1)
        assert x.shape == (batch_size, self.num_classes)
        return x


class MultiLayerPerceptronAdam(MultiLayerPerceptron):
    def __init__(
        self,
        image_size=28,
        num_channels=3,
        num_classes=10,
        *,
        hidden_size_1=128,
        hidden_size_2=256,
        **kwargs,
    ):
        # optimizer
        optimizer_cls = torch.optim.Adam
        optimizer_argnames = ["lr"]
        # scheduler
        scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
        scheduler_argnames = ["gamma"]
        scheduler_interval = "epoch"
        scheduler_add_total_steps = False

        optimizer_kwargs = {k: kwargs.pop(k) for k in optimizer_argnames if k in kwargs}
        scheduler_kwargs = {k: kwargs.pop(k) for k in scheduler_argnames if k in kwargs}
        if kwargs:
            raise TypeError(f"Unexpected keywords {list(kwargs.keys())}")

        super().__init__(
            image_size=image_size,
            num_channels=num_channels,
            num_classes=num_classes,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_interval=scheduler_interval,
            scheduler_add_total_steps=scheduler_add_total_steps,
        )
