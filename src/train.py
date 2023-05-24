#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path

import datamodules
import models
import torch
import wandb
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything
from torchvision import transforms

LOGS_DIR = Path(tempfile.gettempdir()) / "logs"


def train(seed, *, use_wandb=True):
    # set seed
    seed = seed_everything(seed)

    # set data
    datamodule = datamodules.ImagesDataModule(
        # see torchvision.datasets for options
        "FashionMNIST",
        num_channels=1,
        num_classes=10,
        batch_size=256 if torch.cuda.is_available() else 64,
        num_workers=os.cpu_count() - 1,
        train_transforms=[
            transforms.RandomCrop(28),
            # transforms.RandomHorizontalFlip(),
        ],
        eval_transforms=[transforms.CenterCrop(28)],
    )

    # set model
    model = models.FullyConnectedAutoEncoderSGD(
        num_channels=datamodule.num_channels,
        # SGD
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4,
        max_lr=0.1,
        # # Adam
        # lr=0.05,
        # gamma=0.95,
    )
    torch.set_float32_matmul_precision("medium")

    # set logger(s)
    project_name = f"{type(model).__name__.lower()}-{datamodule.dataset_name.lower()}"
    logger = []
    save_dir = LOGS_DIR / project_name
    save_dir.mkdir(exist_ok=True, parents=True)
    if use_wandb:
        if wandb.run is not None:
            wandb.finish()
        wandb_logger = loggers.WandbLogger(project=project_name, save_dir=save_dir)
        logger.append(wandb_logger)

    # set trainer
    trainer = Trainer(
        max_epochs=30,
        logger=logger,
        callbacks=[
            callbacks.RichModelSummary(max_depth=2),
            callbacks.RichProgressBar(),
            callbacks.LearningRateMonitor(logging_interval="step"),
            # callbacks.progress.TQDMProgressBar(refresh_rate=10),
            # callbacks.EarlyStopping("loss/validation"),
        ],
        precision="bf16-mixed",
    )
    trainer.logger.log_hyperparams({"seed": seed})
    trainer.test(model, datamodule=datamodule, verbose=False)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


def main():
    train(seed=None)


if __name__ == "__main__":
    main()
