import itertools

from einops.layers.torch import Rearrange
from torch import nn

from . import base


class FullyConnectedAutoEncoder(base.ImageAutoEncoder):
    def __init__(
        self, image_size=28, num_channels=3, *, hidden_sizes=(64, 4), **kwargs
    ):
        super().__init__(image_size=image_size, num_channels=num_channels, **kwargs)
        self.save_hyperparameters()

        # create the model
        self.save_hyperparameters("hidden_sizes")
        input_size = num_channels * image_size * image_size
        layer_sizes = [input_size, *hidden_sizes]

        encoder_layers = []
        for size_in, size_out in itertools.pairwise(layer_sizes):
            encoder_layers.append(nn.Linear(size_in, size_out))
            encoder_layers.append(nn.ReLU())
        encoder_rearrange = Rearrange(
            "b c h w -> b (c h w)",
            c=self.num_channels,
            h=self.image_size,
            w=self.image_size,
        )
        self.encoder = nn.Sequential(encoder_rearrange, *encoder_layers[:-1], nn.Tanh())

        decoder_layers = []
        for size_in, size_out in itertools.pairwise(layer_sizes[::-1]):
            decoder_layers.append(nn.Linear(size_in, size_out))
            decoder_layers.append(nn.ReLU())
        decoder_rearrange = Rearrange(
            "b (c h w) -> b c h w",
            c=self.num_channels,
            h=self.image_size,
            w=self.image_size,
        )
        self.decoder = nn.Sequential(*decoder_layers[:-1], nn.Tanh(), decoder_rearrange)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FullyConnectedAutoEncoderSGD(
    base.schedulers.OneCycleLR, base.optimizers.SGD, FullyConnectedAutoEncoder
):
    pass
