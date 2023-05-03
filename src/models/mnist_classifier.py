# https://github.com/rubentea16/pl-mnist/blob/master/model.py
# https://docs.ray.io/en/latest/tune/examples/includes/mnist_ptl_mini.html
import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812


class MNISTClassifier(pl.LightningModule):
    def __init__(self, lr, channels=1):
        super().__init__()
        # images are (c, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(channels * 28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        self.lr = lr

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        assert channels == self.channels
        assert width == 28
        assert height == 28
        # (b, c, 28, 28) -> (b, c*28*28)
        x = x.view(batch_size, -1)
        # layer 1 (b, c*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)
        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)
        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)
        # probability distribution over labels
        x = torch.softmax(x, dim=1)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
            "name": "expo_lr",
        }
        return [optimizer], [lr_scheduler]
