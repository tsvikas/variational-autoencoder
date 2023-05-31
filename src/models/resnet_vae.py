import copy  # noqa: I001
import itertools  # noqa: F401
import time
from typing import Callable, Optional, Union  # noqa: F401, UP035

import torch
from torch import nn
from torch import Tensor
from torchvision.models import resnet18  # noqa: F401
import torch.utils.data
import torchvision.datasets
from torchvision import transforms
from tqdm import tqdm

import wandb

from . import base


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, **_kw) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def transposed_conv1x1(
    in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0
) -> nn.ConvTranspose2d:
    """1x1 transposed convolution"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        output_padding=output_padding,
    )


def transposed_conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    _dilation: int = 1,
    output_padding: int = 1,
) -> nn.ConvTranspose2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        output_padding=output_padding,
        groups=groups,
        bias=False,
        # dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,  # noqa: UP007
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,  # noqa: UP007
        output_padding: int = 1,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if inplanes > planes:
            self.conv1 = transposed_conv3x3(
                inplanes, planes, stride, output_padding=output_padding
            )
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualAutoencoder(base.ImageAutoEncoder):
    def __init__(self, bottleneck, layers=[2, 2, 2, 2], **kw):  # noqa: B006
        super().__init__(**kw)
        self.save_hyperparameters()
        self.dilation = 1
        self.inplanes = 4
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(4, layers[0])
        self.layer2 = self._make_layer(8, layers[1], stride=2)
        self.layer3 = self._make_layer(16, layers[2], stride=2)
        self.bottleneck = nn.Linear(1024, bottleneck)
        self.decode_bottleneck = nn.Linear(bottleneck, 1024)
        self.layer3_ = self._make_layer_(
            8, layers[2], stride=2, output_padding=1, downsample_output_padding=1
        )
        self.layer2_ = self._make_layer_(
            4, layers[1], stride=2, output_padding=1, downsample_output_padding=1
        )
        self.layer1_ = self._make_layer_(1, layers[1])
        # self.tconv1 = nn.ConvTranspose2d(
        #     self.inplanes, 1, kernel_size=1, stride=1, padding=0, bias=False
        # )

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        downsample_output_padding: int = 0,
        output_padding: int = 0,
        resample: Callable = conv1x1,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                resample(
                    self.inplanes,
                    planes,
                    stride,
                    output_padding=downsample_output_padding,
                ),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(
            BasicBlock(
                self.inplanes,
                planes,
                stride,
                downsample,
                1,
                64,
                previous_dilation,
                nn.BatchNorm2d,
                output_padding=output_padding,
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    groups=1,
                    base_width=64,
                    dilation=self.dilation,
                    norm_layer=nn.BatchNorm2d,
                    output_padding=output_padding,
                )
            )

        return nn.Sequential(*layers)

    def _make_layer_(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        output_padding: int = 0,
        downsample_output_padding: int = 0,
    ) -> nn.Sequential:
        return self._make_layer(
            planes,
            blocks,
            stride,
            resample=transposed_conv1x1,
            output_padding=output_padding,
            downsample_output_padding=downsample_output_padding,
        )

    def forward(self, x):
        shape = x.shape

        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        image_shape = x.shape
        x = x.reshape(shape[0], -1)
        x = self.bottleneck(x)
        x = self.decode_bottleneck(x)
        x = x.reshape(image_shape)

        x = self.layer3_(x)
        x = self.layer2_(x)
        x = self.layer1_(x)

        # x = self.tconv1(x)

        assert shape == x.shape, (shape, x.shape)
        return x


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))  # noqa: UP032
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                labels = labels.to(device)  # noqa: PLW2901
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    B, _C, H, W = inputs.shape  # noqa: N806
                    assert _C == 1
                    assert outputs.shape == (B, 10)
                    assert labels.shape == (B,)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            wandb.log(
                {"epoch": epoch, phase + "/loss": epoch_loss, phase + "/acc": epoch_acc}
            )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))  # noqa: UP032

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_autoencoder(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e100
    # best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))  # noqa: UP032
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):  # noqa: B007
                # labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    B, _C, H, W = inputs.shape  # noqa: N806
                    assert _C == 1
                    assert outputs.shape == inputs.shape
                    # assert labels.shape == (B,)
                    loss = criterion(outputs.reshape(B, -1), inputs.reshape(B, -1))
                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            wandb.log(
                {
                    "epoch": epoch,
                    phase + "/loss": epoch_loss,
                }  # , phase + "/acc": epoch_acc}
            )

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                # best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                inputs = next(iter(dataloaders[phase]))[0][:8]
                outputs = model(inputs)
                image_array = torch.concat((inputs, outputs))

                wandb.log(
                    {
                        "examples": wandb.Image(
                            image_array, caption="Top: Input, Bottom: Output"
                        )
                    }
                )

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    # print("Best val Acc: {:4f}".format(best_acc))
    print("Best val Loss: {:4f}".format(best_loss))  # noqa: UP032

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    args = {}

    normalize = transforms.Normalize([72.9404 / 255], [90.0212 / 255])
    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                transforms.Lambda(lambda x: x.to(device)),
            ]
        ),
        "val": transforms.Compose(
            [
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
                transforms.Lambda(lambda x: x.to(device)),
            ]
        ),
    }

    train_data = torchvision.datasets.FashionMNIST(
        "fashion-mnist",
        train=True,
        download=True,
        transform=data_transforms["train"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
    )

    test_data = torchvision.datasets.FashionMNIST(
        "fashion-mnist", train=False, transform=data_transforms["val"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=64,
        shuffle=True,
    )

    """
    model = resnet18()
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
    model = model.to(device)
    """
    # wandb.init(config=args, save_code=True)
    model = ResidualAutoencoder(16).to(device)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())
    try:
        model = train_autoencoder(
            model,
            {"train": train_loader, "val": test_loader},
            criterion,
            optimizer,
            num_epochs=5,
        )
        torch.save(model.state_dict(), "weights.pkl")
    finally:
        pass  # wandb.finish()
