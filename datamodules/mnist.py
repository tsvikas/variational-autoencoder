import torchvision
from torchvision.datasets import MNIST as MNISTDataset  # noqa: N811

from .datamodules import ImagesDataModule

# this normalization is taken from https://pytorch-lightning.readthedocs.io/en/1.6.2/starter/core_guide.html
mnist_normalization = torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])


class MNIST(ImagesDataModule):
    dataset_cls = MNISTDataset
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), mnist_normalization]
    )
    val_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), mnist_normalization]
    )
    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), mnist_normalization]
    )
