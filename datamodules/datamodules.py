from typing import Any

import torch
import torch.utils.data
import torchvision
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.seed import isolate_rng


def train_val_split(
    train_length: int,
    val_length: int,
    train_transform,
    val_transform,
    dataset_fn,
    **dataset_kwargs
):
    """load a dataset and split it, using a different transform for train and val"""
    lengths = [train_length, val_length]
    dataset_train = dataset_fn(**dataset_kwargs, transform=train_transform)
    with isolate_rng():
        train_split, _ = torch.utils.data.random_split(dataset_train, lengths)
    dataset_val = dataset_fn(**dataset_kwargs, transform=val_transform)
    _, val_split = torch.utils.data.random_split(dataset_val, lengths)
    return train_split, val_split


class ImagesDataModule(LightningDataModule):
    dataset_cls: Any | type[torchvision.datasets.VisionDataset]
    extra_transforms = None
    mean = None
    std = None

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        val_size: int | float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._val_size = val_size

        # defined in self.setup()
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.train_val_size = None
        self.test_size = None

    @property
    def train_size(self):
        return self.train_val_size - self.val_size

    @property
    def val_size(self):
        return (
            self._val_size
            if self._val_size >= 1
            else int(self._val_size * self.train_val_size)
        )

    def prepare_data(self):
        # download
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    @property
    def normalize_transform(self):
        return torchvision.transforms.Normalize(self.mean, self.std)

    @property
    def test_transforms(self):
        return [torchvision.transforms.ToTensor(), self.normalize_transform]

    @property
    def val_transforms(self):
        return self.test_transforms

    @property
    def train_transforms(self):
        extra_transforms = self.extra_transforms or []
        return [
            *extra_transforms,
            torchvision.transforms.ToTensor(),
            self.normalize_transform,
        ]

    def setup(self, stage=None):
        train_val_dataset = self.dataset_cls(self.data_dir, train=True)
        if self.mean is None:
            self.mean = train_val_dataset.data.mean((0, 1, 2))
        if self.std is None:
            self.std = train_val_dataset.data.std((0, 1, 2))

        self.train_val_size = len(train_val_dataset)
        self.train_set, self.val_set = train_val_split(
            self.train_size,
            self.val_size,
            torchvision.transforms.Compose(self.train_transforms),
            torchvision.transforms.Compose(self.val_transforms),
            self.dataset_cls,
            root=self.data_dir,
            train=True,
            download=False,
        )
        self.test_set = self.dataset_cls(
            root=self.data_dir,
            train=False,
            download=False,
            transform=torchvision.transforms.Compose(self.test_transforms),
        )
        self.test_size = len(self.test_set)

    # added for exploration:
    @property
    def classes(self):
        return self.test_set.classes

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def images_set(self):
        return self.dataset_cls(self.data_dir, train=True, transform=None)

    @property
    def tensors_set(self):
        return self.dataset_cls(
            self.data_dir, train=True, transform=torchvision.transforms.ToTensor()
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
