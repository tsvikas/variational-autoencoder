import os

import torch
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything
from torchvision import transforms

import datamodules
import models


def train(seed=7, *, use_wandb=False):
    seed_everything(seed)

    # set model and data
    datamodule = datamodules.ImagesDataModule(
        "CIFAR10",
        data_dir="./data",
        batch_size=256 if torch.cuda.is_available() else 64,
        num_workers=os.cpu_count() - 1,
        extra_transforms=[
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ],
    )
    model = models.Resnet(lr=0.05)

    # set trainer
    # TODO: wandb.init(save_code=True) -- is it needed?
    logger = [loggers.CSVLogger(save_dir="logs/")]
    if use_wandb:
        project_name = (
            f"{model.__name__.lower()}-{datamodule.dataset_cls.__name__.lower()}"
        )
        logger.append(loggers.WandbLogger(project=project_name))
    trainer = Trainer(
        max_epochs=30,
        devices=1,
        logger=logger,
        callbacks=[
            callbacks.LearningRateMonitor(logging_interval="step"),
            callbacks.progress.TQDMProgressBar(refresh_rate=10),
        ],
        precision="bf16-mixed",
        fast_dev_run=True,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


def main():
    train(seed=7, use_wandb=False)
    # TODO: add wandb.finish()? it'll prevent resuming.


if __name__ == "__main__":
    main()
