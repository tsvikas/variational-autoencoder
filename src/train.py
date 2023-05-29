#!/usr/bin/env python3
import os
import tempfile
import time
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
        # see torchvision.datasets for available datasets
        "FashionMNIST",
        num_channels=1,
        num_classes=10,
        batch_size=512 if torch.cuda.is_available() else 64,
        num_workers=os.cpu_count() - 1,
        train_transforms=[transforms.CenterCrop(28)],
        eval_transforms=[transforms.CenterCrop(28)],
        target_is_self=True,
        noise_transforms=[GaussianNoise(0.1), SaltPepperNoise(0.1, 0.1)],
    )


def get_model(num_channels):
    return models.FullyConnectedAutoEncoder(
        num_channels=num_channels,
        hidden_sizes=(256, 64, 8),
        encoder_last_layer=torch.nn.LayerNorm,
        encoder_last_layer_args=(8,),
        decoder_last_layer=torch.nn.Identity,
        decoder_last_layer_args=(),
        # # SGD
        # optimizer_cls=torch.optim.SGD,
        # optimizer_kwargs=dict(lr=0.1, momentum=0.9, weight_decay=5e-4),
        # AdamW
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
        # OneCycleLR
        # scheduler_cls=torch.optim.lr_scheduler.OneCycleLR,
        # scheduler_kwargs=dict(max_lr=0.1),
        # scheduler_interval="step",
        # scheduler_add_total_steps=True,
        # # Adam
        # optimizer_cls=torch.optim.Adam,
        # optimizer_kwargs=dict(lr=0.05),
        # ExponentialLR
        scheduler_cls=torch.optim.lr_scheduler.ExponentialLR,
        scheduler_kwargs=dict(gamma=0.95),
        scheduler_interval="epoch",
        scheduler_add_total_steps=False,
    )


def train(seed):
    seed = seed_everything(seed)
    datamodule = get_datamodule()
    model = get_model(datamodule.num_channels)
    logger = get_logger(
        project_name=f"{type(model).__name__.lower()}-{datamodule.dataset_name.lower()}"
    )

    # set trainer
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(
        max_epochs=30,
        logger=logger,
        callbacks=[
            callbacks.RichModelSummary(max_depth=2),
            callbacks.RichProgressBar(),
            callbacks.LearningRateMonitor(logging_interval="step"),
            callbacks.EarlyStopping("loss/validation", min_delta=0.001),
        ],
        precision="bf16-mixed",
        enable_model_summary=False,
    )
    trainer.logger.log_hyperparams({"seed": seed})

    # run trainer
    trainer.test(model, datamodule=datamodule, verbose=False)
    t_start = time.time()
    trainer.fit(model, datamodule=datamodule)
    t_total = time.time() - t_start
    print(f"Training took {t_total:.2f} seconds", flush=True)
    trainer.test(model, datamodule=datamodule)


def main():
    train(seed=None)


if __name__ == "__main__":
    main()
