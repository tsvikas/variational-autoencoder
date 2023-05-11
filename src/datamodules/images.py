import tempfile
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.seed import isolate_rng

DATA_DIR = Path(tempfile.gettempdir()) / "images_data"
DATASET_STATS = {
    "MNIST": dict(
        # this normalization is taken from
        # https://pytorch-lightning.readthedocs.io/en/1.6.2/starter/core_guide.html
        mean=[0.1307],
        std=[0.3081],
    ),
    "CIFAR10": dict(
        # this normalization is taken from pl-bolts sourcecode
        mean=[x / 255 for x in [125.3, 123.0, 113.9]],
        std=[x / 255 for x in [63.0, 62.1, 66.7]],
    ),
}


def train_val_split(
    train_length: int,
    val_length: int,
    train_transform,
    val_transform,
    dataset_cls,
    **dataset_kwargs,
):
    """load a dataset and split it, using a different transform for train and val"""
    lengths = [train_length, val_length]
    dataset_train = dataset_cls(**dataset_kwargs, transform=train_transform)
    with isolate_rng():
        train_split, _ = torch.utils.data.random_split(dataset_train, lengths)
    dataset_val = dataset_cls(**dataset_kwargs, transform=val_transform)
    _, val_split = torch.utils.data.random_split(dataset_val, lengths)
    return train_split, val_split


class ImagesDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str | None = None,
        dataset_cls: torchvision.datasets.VisionDataset | None = None,
        num_channels: int = None,
        num_classes: int = None,
        *,
        batch_size: int = None,
        data_dir: str | Path = DATA_DIR,
        train_transforms: list[torch.nn.Module] | None = None,
        eval_transforms: list[torch.nn.Module] | None = None,
        num_workers: int = 1,
        val_size: int | float = 0.2,
    ):
        super().__init__()
        if dataset_name is None and dataset_cls is None:
            raise ValueError("must specify dataset_name or dataset_cls")
        self.dataset_cls = dataset_cls or getattr(
            torchvision.datasets, dataset_name
        )  # type: type[torchvision.datasets.VisionDataset] | type[torchvision.datasets.CIFAR10]
        self.dataset_name = dataset_name or self.dataset_cls.__name__
        self.num_channels = int(num_channels)
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.data_dir = Path(data_dir).as_posix()
        self.train_transforms = train_transforms or []
        self.eval_transforms = eval_transforms or []
        self.num_workers = num_workers
        self._val_size = val_size

        # defined in self.setup()
        self.train_val_size = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    @property
    def train_size(self):
        if self.train_val_size is None:
            raise RuntimeError("train_size is undefined before setup")
        return self.train_val_size - self.val_size

    @property
    def val_size(self):
        if self.train_val_size is None:
            raise RuntimeError("val_size is undefined before setup")
        return (
            self._val_size
            if self._val_size >= 1
            else int(self._val_size * self.train_val_size)
        )

    @property
    def test_size(self):
        if self.train_val_size is None:
            raise RuntimeError("test_set is undefined before setup")
        return len(self.test_set)

    def prepare_data(self):
        # download
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        train_val_dataset = self.dataset_cls(self.data_dir, train=True)
        self.train_val_size = len(train_val_dataset)

        # normalize
        mean, std = calc_mean_and_std(train_val_dataset.data)

        # verify
        saved_mean = DATASET_STATS.get(self.dataset_name, {}).get("mean", None)
        saved_std = DATASET_STATS.get(self.dataset_name, {}).get("std", None)
        if saved_mean is not None:
            torch.testing.assert_close(saved_mean, mean, rtol=1e-3, atol=1e-3)
        if saved_std is not None:
            torch.testing.assert_close(saved_std, std, rtol=1e-3, atol=1e-3)

        normalize_transform = torchvision.transforms.Normalize(mean, std)
        train_transforms = [
            *self.train_transforms,
            torchvision.transforms.ToTensor(),
            normalize_transform,
        ]
        val_transforms = [
            *self.eval_transforms,
            torchvision.transforms.ToTensor(),
            normalize_transform,
        ]
        test_transforms = [
            *self.eval_transforms,
            torchvision.transforms.ToTensor(),
            normalize_transform,
        ]

        self.train_set, self.val_set = train_val_split(
            self.train_size,
            self.val_size,
            torchvision.transforms.Compose(train_transforms),
            torchvision.transforms.Compose(val_transforms),
            self.dataset_cls,
            root=self.data_dir,
            train=True,
            download=False,
        )
        self.test_set = self.dataset_cls(
            root=self.data_dir,
            train=False,
            download=False,
            transform=torchvision.transforms.Compose(test_transforms),
        )
        assert len(self.test_set.classes) == self.num_classes
        assert self.test_set[0][0].shape[0] == self.num_channels

    # added for exploration:
    @property
    def classes(self):
        return self.test_set.classes

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


def calc_mean_and_std(images):
    # shape = (B H W) or (B H W C)
    if images.max() > 1:
        images = images / 255
    images = images / 1.0
    if images.ndim == 3:
        images = images[:, :, :, None]
    if images.ndim != 4:
        raise ValueError("ndim not in [3, 4]")
    mean = images.mean((0, 1, 2)).tolist()
    std = images.std((0, 1, 2)).tolist()
    return mean, std
