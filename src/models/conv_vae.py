from collections.abc import Callable

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from . import base
from .base import VAEOutput

ActivationT = Callable[[torch.Tensor], torch.Tensor]


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_fn: ActivationT,
    ):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity(),
        )

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.act = act_fn
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.bn(x)
        x = x + residual
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_fn: ActivationT = nn.functional.gelu,
        output_padding: int = 1,
    ):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
            if in_channels != out_channels
            else nn.Identity(),
            nn.Upsample(scale_factor=2),
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
        residual = self.shortcut(x)
        x = self.act(self.conv_t1(x))
        x = self.act(self.conv_t2(x))
        x = self.bn(x)
        x = x + residual
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        channels: tuple[int],
        latent_dim: int,
        act_fn: ActivationT = nn.functional.gelu,
        latent_act_fn: type[nn.Module] = nn.Identity,
        first_kernel_size: int = 7,
        image_size: int = 32,
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
        self.image_size = image_size
        self.bottleneck_size = image_size // 2 // 2 // 2 // 2

        self.conv = nn.Conv2d(
            num_input_channels,
            channels[0],
            kernel_size=first_kernel_size,
            padding=first_kernel_size // 2,
            stride=2,
            bias=False,
        )

        self.down1 = DownBlock(channels[0], channels[1], act_fn)
        self.down2 = DownBlock(channels[1], channels[2], act_fn)
        self.down3 = DownBlock(channels[2], channels[3], act_fn)

        self.flatten = Rearrange(
            "b c h w -> b (c h w)",
            h=self.bottleneck_size,
            w=self.bottleneck_size,
            c=channels[3],
        )
        self.mu = nn.Linear(
            self.bottleneck_size * self.bottleneck_size * channels[3], latent_dim
        )
        self.log_var = nn.Linear(
            self.bottleneck_size * self.bottleneck_size * channels[3], latent_dim
        )
        self.latent_act = latent_act_fn()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.conv(x))
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.flatten(x)
        x = self.latent_act(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


# %%
class Decoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        channels: tuple[int],
        latent_dim: int,
        act_fn: ActivationT = nn.functional.gelu,
        first_kernel_size: int = 7,
        image_size: int = 32,
    ):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.act = act_fn
        self.image_size = image_size
        self.bottleneck_size = image_size // 2 // 2 // 2 // 2

        self.linear = nn.Linear(
            latent_dim, self.bottleneck_size * self.bottleneck_size * channels[3]
        )
        self.reshape = Rearrange(
            "b (c h w) -> b c h w", h=self.bottleneck_size, w=self.bottleneck_size
        )
        self.up1 = UpBlock(channels[3], channels[2], act_fn)
        self.up2 = UpBlock(channels[2], channels[1], act_fn)
        self.up3 = UpBlock(channels[1], channels[0], act_fn)
        self.conv = nn.ConvTranspose2d(
            channels[0],
            num_input_channels,
            kernel_size=first_kernel_size,
            padding=first_kernel_size // 2,
            stride=2,
            output_padding=1,
            bias=False,
        )

    def forward(self, x):
        x = self.act(self.linear(x))
        x = self.reshape(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv(x)
        return x


class ConvVAE(base.ImageAutoEncoder):
    def __init__(
        self,
        channels: tuple[int, int, int, int] = (16, 16, 16, 16),
        latent_dim: int = 8,
        encoder_class: type[nn.Module] = Encoder,
        decoder_class: type[nn.Module] = Decoder,
        num_channels: int = 1,
        latent_noise: float = 0.0,
        first_kernel_size: int = 5,
        image_size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs, num_channels=num_channels, image_size=image_size)
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(
            num_channels,
            channels,
            latent_dim,
            first_kernel_size=first_kernel_size,
            image_size=image_size,
        )
        self.decoder = decoder_class(
            num_channels,
            channels,
            latent_dim,
            first_kernel_size=first_kernel_size,
            image_size=image_size,
        )

        self.latent_dim = latent_dim
        self.num_input_channels = num_channels
        self.latent_noise = latent_noise

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """The forward function takes in an image and returns the reconstructed image."""
        mu, log_var_2 = self.encoder(x)
        # z is the latent representation
        z = self.reparameterize(mu, log_var_2)
        x_hat = self.decoder(z)
        return VAEOutput(x_hat=x_hat, mu=mu, log_var_2=log_var_2)
