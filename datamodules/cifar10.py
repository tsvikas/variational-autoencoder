import torchvision
from torchvision.datasets import CIFAR10 as CIFAR10Dataset  # noqa: N811

from .datamodules import ImagesDataModule


class CIFAR10(ImagesDataModule):
    dataset_cls = CIFAR10Dataset
    # this normalization is taken from pl-bolts.
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    extra_transforms = [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
    ]
