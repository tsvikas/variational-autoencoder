import torch.optim
from torch import nn

from .modules import ImageAutoEncoder


class FullyConnectedAutoEncoder(ImageAutoEncoder):
    def __init__(
        self,
        image_size=28,
        num_channels=3,
        *,
        hidden_size_1=256,
        hidden_size_2=64,
        hidden_size_3=16,
    ):
        super().__init__(image_size=image_size, num_channels=num_channels)
        # create the model
        self.save_hyperparameters("hidden_size_1", "hidden_size_2", "hidden_size_3")
        input_size = num_channels * image_size * image_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size_3, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, input_size),
            nn.Tanh(),
        )

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
            optimizer_cls=torch.optim.Adam,
            optimizer_hparams={"lr": 0.05},
            scheduler_cls=torch.optim.lr_scheduler.ExponentialLR,
            scheduler_interval="epoch",
            scheduler_hparams={"gamma": 0.95},
            add_total_steps=False,
        )
