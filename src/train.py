#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path

import datamodules
import models
import torch
import wandb
from datamodules.noise import GaussianNoise, SaltPepperNoise
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything
from torchvision import transforms

LOGS_DIR = Path(tempfile.gettempdir()) / "logs"


def get_logger(project_name: str):
    # save dir
    save_dir = LOGS_DIR / project_name
    save_dir.mkdir(exist_ok=True, parents=True)
    logger = []
    # wandb
    if wandb.run is not None:
        wandb.finish()
    wandb_logger = loggers.WandbLogger(project=project_name, save_dir=save_dir)
    logger.append(wandb_logger)
    # return
    return logger


def get_datamodule():
    return datamodules.ImagesDataModule(
        # see torchvision.datasets for options
        "FashionMNIST",
        num_channels=1,
        num_classes=10,
        batch_size=256 if torch.cuda.is_available() else 64,
        num_workers=os.cpu_count() - 1,
        train_transforms=[transforms.CenterCrop(28)],
        eval_transforms=[transforms.CenterCrop(28)],
        target_is_self=True,
        noise_transforms=[GaussianNoise(0.05), SaltPepperNoise(0.05, 0.05)],
    )


def train(seed):
    # set seed
    seed = seed_everything(seed)

    # set data
    datamodule = get_datamodule()

    # set model
    model = models.FullyConnectedAutoEncoderSGD(
        num_channels=datamodule.num_channels,
        hidden_sizes=(256, 64, 8),
        encoder_last_layer=torch.nn.LayerNorm,
        encoder_last_layer_args=(8,),
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
    logger = get_logger(
        project_name=f"{type(model).__name__.lower()}-{datamodule.dataset_name.lower()}"
    )

    # set trainer
    trainer = Trainer(
        max_epochs=30,
        logger=logger,
        callbacks=[
            callbacks.RichModelSummary(max_depth=2),
            callbacks.RichProgressBar(),
            callbacks.LearningRateMonitor(logging_interval="step"),
            # callbacks.EarlyStopping("loss/validation"),
        ],
        precision="bf16-mixed",
        enable_model_summary=False,
        fast_dev_run=True,
    )
    trainer.logger.log_hyperparams({"seed": seed})
    trainer.test(model, datamodule=datamodule, verbose=False)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


def main():
    train(seed=None)


if __name__ == "__main__":
    main()
