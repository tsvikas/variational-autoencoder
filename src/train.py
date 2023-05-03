#!/usr/bin/env python3
import os

import datamodules
import models
import torch
import wandb
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything
from torchvision import transforms


def train(seed, *, use_wandb=True):
    seed_everything(seed)

    # set model and data
    datamodule = datamodules.ImagesDataModule(
        "FashionMNIST",
        num_classes=10,
        num_channels=1,
        data_dir="../data",
        batch_size=256 if torch.cuda.is_available() else 64,
        num_workers=os.cpu_count() - 1,
        extra_transforms=[
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ],
    )
    model = models.Resnet(
        lr=0.05,
        num_channels=datamodule.num_channels,
        num_classes=datamodule.num_classes,
    )
    torch.set_float32_matmul_precision("medium")

    # set trainer
    project_name = f"{type(model).__name__.lower()}-{datamodule.dataset_name.lower()}"
    logger = [loggers.CSVLogger(save_dir=f"logs/{project_name}")]
    if use_wandb:
        if wandb.run is not None:
            wandb.finish()
        wandb_logger = loggers.WandbLogger(project=project_name, save_code=True)
        logger.append(wandb_logger)
    trainer = Trainer(
        max_epochs=30,
        devices=1,
        logger=logger,
        callbacks=[
            callbacks.LearningRateMonitor(logging_interval="step"),
            callbacks.progress.TQDMProgressBar(refresh_rate=10),
            callbacks.EarlyStopping("val_loss"),
        ],
        precision="bf16-mixed",
    )
    trainer.test(model, datamodule=datamodule, verbose=False)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


def main():
    train(seed=7)
    # TODO: add wandb.finish()? it'll prevent resuming.


if __name__ == "__main__":
    main()
