from collections.abc import Callable

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from . import base


class Encoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
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
        c_hid = base_channel_size

        # 28x28 => 14x14
        self.conv1 = nn.Conv2d(
            num_input_channels, c_hid, kernel_size=3, padding=1, stride=2
        )
        self.conv2 = nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1)

        # 14x14 => 7x7
        self.conv3 = nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1)

        # 7x7 => 4x4
        self.conv5 = nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2)

        self.flatten = nn.Flatten()
        self.ln = nn.LayerNorm(2 * 16 * c_hid)
        self.linear = nn.Linear(2 * 16 * c_hid, latent_dim)
        self.latent_act = latent_act_fn()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.flatten(x)
        x = self.ln(x)
        x = self.linear(x)
        x = self.latent_act(x)
        return x


# %%
class Decoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
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
        c_hid = base_channel_size
        self.linear = nn.Linear(latent_dim, 2 * 16 * c_hid)
        self.ln = nn.LayerNorm(2 * 16 * c_hid, elementwise_affine=True)
        self.reshape = Rearrange("b (c h w) -> b c h w", h=4, w=4)
        self.act = act_fn

        # 4x4 => 7x7
        self.convt1 = nn.ConvTranspose2d(
            2 * c_hid,
            2 * c_hid,
            kernel_size=3,
            output_padding=0,  # NOTE! This was modified to support 28x28 images, instead of 32x32
            padding=1,
            stride=2,
        )
        self.conv1 = nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1)

        # 7x7 => 14x14
        self.convt2 = nn.ConvTranspose2d(
            2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
        )
        self.conv2 = nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1)

        # 14x14 => 28x28
        self.convt3 = nn.ConvTranspose2d(
            c_hid,
            c_hid,
            kernel_size=3,
            output_padding=1,
            padding=1,
            stride=2,
        )
        self.conv3 = nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1)

        # 28x28 => 56x56
        self.convt4 = nn.ConvTranspose2d(
            c_hid,
            c_hid,
            kernel_size=3,
            output_padding=1,
            padding=1,
            stride=2,
        )
        self.conv4 = nn.Conv2d(
            c_hid, num_input_channels, kernel_size=3, padding=1, stride=2
        )

        # self.conv4 = nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.act(self.linear(x))
        x = self.ln(x)
        x = self.reshape(x)
        x = self.act(self.convt1(x))
        x = self.act(self.conv1(x))
        x = self.act(self.convt2(x))
        x = self.act(self.conv2(x))
        x = self.act(self.convt3(x))
        x = self.act(self.conv3(x))
        x = self.act(self.convt4(x))
        x = self.conv4(x)
        return x


# %%
class ConvAutoencoder(base.ImageAutoEncoder):
    def __init__(
        self,
        base_channel_size: int = 6,
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
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)

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
