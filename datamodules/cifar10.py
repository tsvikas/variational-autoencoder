import torchvision
from torchvision.datasets import CIFAR10 as CIFAR10Dataset  # noqa: N811

from .datamodules import ImagesDataModule

# this normalization is taken from pl-bolts.
# it uses the mean/std of the cifar10 train dataset,
# which can be calculated using:
# CIFAR10Dataset(train=True).data.mean((0, 1, 2)).round(1)
# CIFAR10Dataset(train=True).data.std((0, 1, 2)).round(1)
cifar10_normalization = torchvision.transforms.Normalize(
    mean=[x / 255 for x in [125.3, 123.0, 113.9]],
    std=[x / 255 for x in [63.0, 62.1, 66.7]],
)


class CIFAR10(ImagesDataModule):
    dataset_cls = CIFAR10Dataset
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization,
        ]
    )
    val_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), cifar10_normalization]
    )
    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), cifar10_normalization]
    )
