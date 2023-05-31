#!/usr/bin/env python3
import os
import tempfile
import time
from pathlib import Path

import torch
import wandb
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything
from torchvision import transforms

import datamodules
import models
from datamodules import noise

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
        # see torchvision.datasets for available datasets
        "FashionMNIST",
        num_channels=1,
        num_classes=10,
        batch_size=2048,
        num_workers=0 if torch.backends.mps.is_available() else os.cpu_count() - 1,
        train_transforms=[
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
        ],
        eval_transforms=[
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
        ],
        target_is_self=True,
        noise_transforms=[
            transforms.RandomApply([transforms.RandomErasing()], p=0.5),
            transforms.RandomApply([noise.SaltPepperNoise(0.1, 0.1)], p=0.5),
            transforms.RandomApply([noise.GaussianNoise(0.1)], p=0.5),
        ],
    )


def get_model(num_channels):
    return models.ConvAutoencoder(
        latent_dim=8,
        image_size=28,
        latent_noise=0.1,
        num_channels=num_channels,
        # # FullyConnectedAutoEncoder
        # hidden_sizes=(256, 64, 8),
        # encoder_last_layer=torch.nn.LayerNorm,
        # encoder_last_layer_args=(8,),
        # decoder_last_layer=torch.nn.Identity,
        # decoder_last_layer_args=(),
        # # SGD
        # optimizer_cls=torch.optim.SGD,
        # optimizer_kwargs=dict(lr=0.1, momentum=0.9, weight_decay=5e-4),
        # # AdamW
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=0.01),
        # # ReduceLROnPlateau
        # scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        # scheduler_kwargs=dict(patience=1, threshold=0.05, factor=0.1),
        # scheduler_monitor="loss/training",
        # scheduler_interval="epoch",
        # # LambdaLR
        # optimizer_kwargs=dict(lr=1),
        # scheduler_cls=torch.optim.lr_scheduler.LambdaLR,
        # scheduler_kwargs=dict(lr_lambda=lambda step: min(0.01, (step+1e-8)**-0.5)),
        # scheduler_interval="step",
        # # OneCycleLR
        # scheduler_cls=torch.optim.lr_scheduler.OneCycleLR,
        # scheduler_kwargs=dict(max_lr=0.1),
        # scheduler_interval="step",
        # scheduler_add_total_steps=True,
        # # Adam
        # optimizer_cls=torch.optim.Adam,
        # optimizer_kwargs=dict(lr=0.05),
        # # ExponentialLR
        # scheduler_cls=torch.optim.lr_scheduler.ExponentialLR,
        # scheduler_kwargs=dict(gamma=0.95),
        # scheduler_interval="epoch",
        # scheduler_add_total_steps=False,
    )


def train(seed):
    seed = seed_everything(seed)
    datamodule = get_datamodule()
    model = get_model(datamodule.num_channels)

    # trainer settings
    max_epochs = 30
    trainer_callbacks = [
        callbacks.EarlyStopping("loss/validation", min_delta=0.001),
    ]

    # set precision
    # torch.set_float32_matmul_precision("medium")
    # precision = "bf16-mixed"
    precision = 16

    # fast_dev_run, to prevent logging of failed runs
    trainer_fast = Trainer(
        accelerator="auto",
        fast_dev_run=True,
        enable_model_summary=False,
        enable_progress_bar=False,
        precision=precision,
        logger=False,
        callbacks=[
            callbacks.RichModelSummary(max_depth=2),
        ],
    )
    trainer_fast.fit(model, datamodule=datamodule)

    # set trainer
    logger = get_logger(
        project_name=f"{type(model).__name__.lower()}-{datamodule.dataset_name.lower()}"
    )
    trainer = Trainer(
        accelerator="auto",
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            callbacks.RichProgressBar(),
            callbacks.LearningRateMonitor(logging_interval="step"),
            *trainer_callbacks,
        ],
        precision=precision,
        enable_model_summary=False,
        log_every_n_steps=5,
    )
    trainer.logger.log_hyperparams({"seed": seed})

    # run trainer
    trainer.test(model, datamodule=datamodule, verbose=False)
    t_start = time.time()
    trainer.fit(model, datamodule=datamodule)
    t_total = time.time() - t_start
    trainer.logger.log_metrics({"trainer/total_time": t_total})
    trainer.test(model, datamodule=datamodule)


def main():
    train(seed=None)


if __name__ == "__main__":
    main()
