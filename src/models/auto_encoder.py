import itertools

import torch
from einops.layers.torch import Rearrange
from torch import nn

from . import base


class FullyConnectedAutoEncoder(base.ImageAutoEncoder):
    def __init__(
        self,
        hidden_sizes=(64, 4),
        encoder_last_layer=nn.Identity,
        encoder_last_layer_args=(),
        image_size=28,
        num_channels=3,
        *,
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
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_interval=scheduler_interval,
            scheduler_add_total_steps=scheduler_add_total_steps,
        )
        self.save_hyperparameters()

        # create the model
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
        self.encoder = nn.Sequential(
            encoder_rearrange,
            *encoder_layers[:-1],
            encoder_last_layer(*encoder_last_layer_args),
        )

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


class FullyConnectedAutoEncoderSGD(FullyConnectedAutoEncoder):
    def __init__(
        self,
        hidden_sizes=(64, 4),
        encoder_last_layer=nn.Identity,
        encoder_last_layer_args=(),
        image_size=28,
        num_channels=3,
        **kwargs,
    ):
        # optimizer
        optimizer_cls = torch.optim.SGD
        optimizer_argnames = ["lr", "momentum", "weight_decay"]
        # scheduler
        scheduler_cls = torch.optim.lr_scheduler.OneCycleLR
        scheduler_argnames = ["max_lr"]
        scheduler_interval = "step"
        scheduler_add_total_steps = True

        optimizer_kwargs = {k: kwargs.pop(k) for k in optimizer_argnames if k in kwargs}
        scheduler_kwargs = {k: kwargs.pop(k) for k in scheduler_argnames if k in kwargs}
        if kwargs:
            raise TypeError(f"Unexpected keywords {list(kwargs.keys())}")

        super().__init__(
            hidden_sizes=hidden_sizes,
            encoder_last_layer=encoder_last_layer,
            encoder_last_layer_args=encoder_last_layer_args,
            image_size=image_size,
            num_channels=num_channels,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_interval=scheduler_interval,
            scheduler_add_total_steps=scheduler_add_total_steps,
        )
