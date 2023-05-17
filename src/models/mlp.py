# https://github.com/rubentea16/pl-mnist/blob/master/model.py
# https://docs.ray.io/en/latest/tune/examples/includes/mnist_ptl_mini.html
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .classifier import NLLClassifierWithOptimizer


class MultiLayerPerceptron(NLLClassifierWithOptimizer):
    def __init__(
        self,
        image_size=28,
        num_channels=3,
        num_classes=10,
        *,
        hidden_size_1=128,
        hidden_size_2=256,
        lr=0.05,
        gamma=0.95
    ):
        model_hparams = {"hidden_size_1": hidden_size_1, "hidden_size_2": hidden_size_2}
        optimizer_hparams = {"lr": lr}
        scheduler_hparams = {"gamma": gamma}
        self.save_hyperparameters(model_hparams)
        super().__init__(
            image_size=image_size,
            num_channels=num_channels,
            num_classes=num_classes,
            optimizer_name_or_cls="Adam",
            optimizer_hparams=optimizer_hparams,
            scheduler_name_or_cls="ExponentialLR",
            scheduler_hparams=scheduler_hparams,
            scheduler_interval="epoch",
            add_total_steps=False,
        )
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
