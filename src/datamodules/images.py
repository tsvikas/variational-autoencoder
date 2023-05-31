import tempfile
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.seed import isolate_rng

DATA_DIR = Path(tempfile.gettempdir()) / "images_data"


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
    with isolate_rng():
        dataset_train = dataset_cls(**dataset_kwargs, transform=train_transform)
        train_split, _ = torch.utils.data.random_split(dataset_train, lengths)
    with isolate_rng():
        dataset_val = dataset_cls(**dataset_kwargs, transform=val_transform)
        _, val_split = torch.utils.data.random_split(dataset_val, lengths)
    # repeat to consume the random state
    dataset = dataset_cls(**dataset_kwargs)
    torch.utils.data.random_split(dataset, lengths)
    return train_split, val_split


def parse_name_or_cls(name_or_cls: str | type[object], namespace):
    if isinstance(name_or_cls, str):
        name = name_or_cls
        cls = getattr(namespace, name)
    else:
        cls = name_or_cls
        name = cls.__name__
    return name, cls


class ImagesDataModule(LightningDataModule):
    """
    Convert between torchvision.datasets.VisionDataset and LightningDataModule
    and allow setting different transforms for train / eval
    """

    def __init__(
        self,
        dataset_name_or_cls: str
        | type[torchvision.datasets.VisionDataset]
        | type[torchvision.datasets.MNIST],
        num_channels: int,
        num_classes: int,
        *,
        batch_size: int = 1,
        data_dir: str | Path = DATA_DIR,
        train_transforms: list[torch.nn.Module] | None = None,
        eval_transforms: list[torch.nn.Module] | None = None,
        num_workers: int = 1,
        val_size_or_frac: int | float = 0.2,
        target_is_self=False,
        noise_transforms: list[torch.nn.Module] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name, self.dataset_cls = parse_name_or_cls(
            dataset_name_or_cls, torchvision.datasets
        )
        if not issubclass(self.dataset_cls, torchvision.datasets.VisionDataset):
            raise ValueError(  # noqa: TRY004
                f"{self.dataset_cls} is not a subclass of torchvision.datasets.VisionDataset"
            )
        self.num_channels = int(num_channels)
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.data_dir = Path(data_dir).as_posix()
        self.train_transforms = train_transforms or []
        self.eval_transforms = eval_transforms or []
        self.num_workers = num_workers
        self.val_size_or_frac = val_size_or_frac
        self.target_is_self = target_is_self
        self.noise_transforms = noise_transforms or []

        # defined in self.setup()
        self.train_val_size = None
        self.normalize_transform = None
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
            int(self.val_size_or_frac)
            if self.val_size_or_frac >= 1
            else int(self.val_size_or_frac * self.train_val_size)
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
        self.normalize_transform = torchvision.transforms.Normalize(mean, std)

        # set transforms
        train_transforms = [
            *self.train_transforms,
            torchvision.transforms.ToTensor(),
            self.normalize_transform,
        ]
        val_transforms = [
            *self.eval_transforms,
            torchvision.transforms.ToTensor(),
            self.normalize_transform,
        ]
        test_transforms = [
            *self.eval_transforms,
            torchvision.transforms.ToTensor(),
            self.normalize_transform,
        ]

        # create dataset
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
        if self.target_is_self:
            self.train_set = TransformedSelfDataset(
                self.train_set, transforms=self.noise_transforms
            )
            self.val_set = TransformedSelfDataset(
                self.val_set, transforms=self.noise_transforms
            )
            self.test_set = TransformedSelfDataset(
                self.test_set, transforms=self.noise_transforms
            )

        # verify num_classes and num_channels
        if (num_classes := len(self.test_set.classes)) != self.num_classes:
            raise ValueError(
                f"{type(self).__name__} should be created with {num_classes=}"
            )
        if (num_channels := self.test_set[0][0].shape[0]) != self.num_channels:
            raise ValueError(
                f"{type(self).__name__} should be created with {num_channels=}"
            )

    # added for exploration:
    @property
    def classes(self):
        if self.test_set is None:
            raise RuntimeError("classes is undefined before setup()")
        return self.test_set.classes

    # added for exploration:
    def dataset(
        self,
        *,
        train=True,
        transforms=None,
        to_tensor=False,
        normalize=False,
        tensor_transforms=None,
    ):
        if transforms is None:
            transforms = []
        if to_tensor:
            transforms.append(torchvision.transforms.ToTensor())
            if normalize:
                transforms.append(self.normalize_transform)
            if tensor_transforms is not None:
                transforms.extend(tensor_transforms)
        else:
            if normalize:
                raise ValueError("can't normalize without converting to tensor")
            if tensor_transforms:
                raise ValueError(
                    "can't use tensor_transforms without converting to tensor"
                )
        transform = torchvision.transforms.Compose(transforms) if transforms else None
        return self.dataset_cls(self.data_dir, train=train, transform=transform)

    # functions needed for LightningDataModule
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # we can use a x2 batch_size in validation and testing,
    # because we don't have gradients
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def calc_mean_and_std(images):
    # images.shape = (B H W) or (B H W C)
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


class TransformedSelfDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms=None):
        super().__init__()
        self.dataset = dataset
        transforms = transforms or []
        self.transform = torchvision.transforms.Compose(transforms)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return self.transform(image), image

    def __len__(self):
        return len(self.dataset)

    @property
    def classes(self):
        return self.dataset.classes
