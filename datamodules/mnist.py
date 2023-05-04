from torchvision.datasets import MNIST as MNISTDataset  # noqa: N811
from torchvision.datasets import FashionMNIST as FashionMNISTDataset

from .datamodules import ImagesDataModule


class MNIST(ImagesDataModule):
    dataset_cls = MNISTDataset
    # this normalization is taken from https://pytorch-lightning.readthedocs.io/en/1.6.2/starter/core_guide.html
    mean = [0.1307]
    std = [0.3081]


class FashionMNIST(ImagesDataModule):
    dataset_cls = FashionMNISTDataset
