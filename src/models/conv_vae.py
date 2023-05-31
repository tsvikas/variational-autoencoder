from collections.abc import Callable

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from . import base


class DownBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, act_fn: type[nn.Module] = nn.GELU
    ):
        super().__init__()
        self.downsample = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=2, padding=0
        )

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.act = act_fn
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.bn(x + residual)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_fn: type[nn.Module] = nn.GELU,
        output_padding: int = 0,
        size=None,
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2) if size is None else nn.Upsample(size=size),
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            ),
        )
        self.conv_t1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=output_padding,
        )
        self.conv_t2 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_fn

    def forward(self, x):
        residual = self.upsample(x)
        x = self.act(self.conv_t1(x))
        x = self.act(self.conv_t2(x))
        x = self.bn(x + residual)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        channels: tuple[int],
        latent_dim: int,
        act_fn: Callable = nn.functional.gelu,
        latent_act_fn: type[nn.Module] = nn.Tanh,
    ):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.act = act_fn

        # 28x28 => 14x14
        self.down1 = DownBlock(num_input_channels, channels[0], act_fn)
        # 14x14 => 7x7
        self.down2 = DownBlock(channels[0], channels[1], act_fn)
        # 7x7 => 4x4
        self.down3 = DownBlock(channels[1], channels[2], act_fn)

        self.flatten = nn.Flatten()
        # self.ln = nn.LayerNorm(2 * 16 * c_hid)
        self.linear = nn.Linear(4 * 4 * channels[2], latent_dim)
        self.latent_act = latent_act_fn()

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.flatten(x)
        # x = self.ln(x)
        x = self.linear(x)
        x = self.latent_act(x)
        return x


# %%
class Decoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        channels: tuple[int],
        latent_dim: int,
        act_fn: Callable = nn.functional.gelu,
    ):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()

        self.linear = nn.Linear(latent_dim, 4 * 4 * channels[2])
        self.ln = nn.LayerNorm(4 * 4 * channels[2])
        self.reshape = Rearrange("b (c h w) -> b c h w", h=4, w=4)
        self.act = act_fn

        self.up1 = UpBlock(channels[2], channels[1], act_fn, size=(7, 7))
        self.up2 = UpBlock(channels[1], channels[0], act_fn, output_padding=1)
        self.up3 = UpBlock(channels[0], num_input_channels, act_fn, output_padding=1)

    def forward(self, x):
        x = self.act(self.linear(x))
        x = self.ln(x)
        x = self.reshape(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x


# %%
class ConvAutoencoder(base.ImageAutoEncoder):
    def __init__(
        self,
        channels: tuple[int] = (16, 16, 32),
        latent_dim: int = 8,
        encoder_class: type[nn.Module] = Encoder,
        decoder_class: type[nn.Module] = Decoder,
        num_input_channels: int = 1,
        width: int = 28,
        height: int = 28,
        latent_noise: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, channels, latent_dim)
        self.decoder = decoder_class(num_input_channels, channels, latent_dim)

        self.latent_dim = latent_dim
        self.num_input_channels = num_input_channels
        self.width = width
        self.height = height
        self.latent_noise = latent_noise

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        if self.training and self.latent_noise > 0.0:
            # Add some noise to the latent representation
            z = z + torch.randn_like(z) * self.latent_noise
        # z is the latent representation
        x_hat = self.decoder(z)
        return x_hat
