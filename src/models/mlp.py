# https://github.com/rubentea16/pl-mnist/blob/master/model.py
# https://docs.ray.io/en/latest/tune/examples/includes/mnist_ptl_mini.html
import torch.nn.functional as F  # noqa: N812
import torch.optim
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
    ):
        super().__init__(
            image_size=image_size, num_channels=num_channels, num_classes=num_classes
        )
        # create the model
        self.save_hyperparameters("hidden_size_1", "hidden_size_2")
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

    def configure_optimizers(self):
        return self.create_optimizers(
            optimizer_cls=torch.optim.Adam,
            optimizer_hparams={"lr": 0.05},
            scheduler_cls=torch.optim.lr_scheduler.ExponentialLR,
            scheduler_interval="epoch",
            scheduler_hparams={"gamma": 0.95},
            add_total_steps=False,
        )
