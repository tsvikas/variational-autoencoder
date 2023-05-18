import itertools

import torch.optim
from torch import nn

from .modules import ImageAutoEncoder


class FullyConnectedAutoEncoder(ImageAutoEncoder):
    def __init__(
        self,
        image_size=28,
        num_channels=3,
        *,
        hidden_sizes=(256, 64, 16),
    ):
        super().__init__(image_size=image_size, num_channels=num_channels)
        # create the model
        self.save_hyperparameters("hidden_sizes")
        input_size = num_channels * image_size * image_size
        layer_sizes = [input_size, *hidden_sizes]

        encoder_layers = []
        for size_in, size_out in itertools.pairwise(layer_sizes):
            encoder_layers.append(nn.Linear(size_in, size_out))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers[:-1])

        decoder_layers = []
        for size_in, size_out in itertools.pairwise(layer_sizes[::-1]):
            decoder_layers.append(nn.Linear(size_in, size_out))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers[:-1], nn.Tanh())

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert channels == self.num_channels
        assert height == width == self.image_size
        # (b, c, 28, 28) -> (b, c*28*28)
        x = x.view(batch_size, -1)
        # -> (b, 16)
        x = self.encoder(x)
        # -> (b, c*28*28)
        x = self.decoder(x)
        # -> (b, c, 28, 28)
        x = x.view(batch_size, channels, height, width)
        return x

    def configure_optimizers(self):
        return self.create_optimizers(
            optimizer_cls=torch.optim.SGD,
            optimizer_hparams={"lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4},
            scheduler_cls=torch.optim.lr_scheduler.OneCycleLR,
            scheduler_hparams={"max_lr": 0.1},
            scheduler_interval="step",
            add_total_steps=True,
        )
