#!/usr/bin/env python3
import os
import tempfile
import time
import warnings
from enum import Enum
from pathlib import Path

import torch
import typer
import wandb
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything
from torchvision import transforms

import datamodules
import models
from datamodules import noise

app = typer.Typer(pretty_exceptions_enable=False)

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*but CUDA is not available.*")
warnings.filterwarnings(
    "ignore", ".*is supported for historical reasons but its usage is discouraged.*"
)
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


def get_datamodule(dataset: str, batch_size: int = 512):
    return datamodules.ImagesDataModule(
        # see torchvision.datasets for available datasets
        dataset,
        num_channels=1,
        num_classes=10,
        batch_size=batch_size,
        num_workers=os.cpu_count() - 1,
        train_transforms=[
            # transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(32),
        ],
        eval_transforms=[
            # transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(32),
        ],
        target_is_self=True,
        noise_transforms=[
            transforms.RandomApply([transforms.RandomErasing()], p=0.5),
            transforms.RandomApply([noise.SaltPepperNoise(0.05, 0.05)], p=0.5),
            transforms.RandomApply([noise.GaussianNoise(0.05)], p=0.5),
        ],
    )


def get_model(
    num_channels: int,
    latent_dim: int = 32,
    latent_noise: float = 0.1,
    channels: tuple[int, int, int, int] = (16, 16, 32, 32),
    kl_weight=0.005,
):
    return models.ConvVAE(
        latent_dim=latent_dim,
        image_size=32,
        latent_noise=latent_noise,
        num_channels=num_channels,
        channels=channels,
        kl_weight=kl_weight,
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
        optimizer_kwargs=dict(lr=0.0003),
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


class AvailableDatasets(str, Enum):
    FashionMNIST = "FashionMNIST"
    KMNIST = "KMNIST"


@app.command()
def train(
    seed: int = 42,
    max_epochs: int = 50,
    latent_dim: int = 32,
    latent_noise: float = 0.1,
    channels: tuple[int, int, int, int] = (32, 64, 128, 256),
    checkpoint_path: str = None,
    batch_size: int = 2048,
    kl_weight: float = 0.005,
    dataset: AvailableDatasets = AvailableDatasets.FashionMNIST,
):
    seed = seed_everything(seed)
    datamodule = get_datamodule(batch_size=batch_size, dataset=dataset.value)
    model = get_model(
        num_channels=datamodule.num_channels,
        latent_dim=latent_dim,
        latent_noise=latent_noise,
        channels=channels,
        kl_weight=kl_weight,
    )

    # trainer settings
    trainer_callbacks = [
        # callbacks.EarlyStopping("loss/validation", min_delta=0.0, patience=10),
        # callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
    ]

    # set precision
    precision = 16
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        precision = "16-mixed"

    # fast_dev_run, to prevent logging of failed runs
    trainer_fast = Trainer(
        accelerator="auto",
        fast_dev_run=True,
        enable_model_summary=False,
        enable_progress_bar=False,
        precision=precision,
        logger=False,
        callbacks=[
            callbacks.RichModelSummary(max_depth=4),
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
            callbacks.LearningRateMonitor(logging_interval="step", log_momentum=True),
            *trainer_callbacks,
        ],
        precision=precision,
        enable_model_summary=False,
        log_every_n_steps=5 if len(datamodule.train_dataloader()) > 5 else 1,
    )
    trainer.logger.log_hyperparams({"seed": seed})

    # run trainer
    trainer.test(model, datamodule=datamodule, verbose=False)
    t_start = time.time()
    trainer.fit(
        model, datamodule=datamodule, ckpt_path=checkpoint_path and str(checkpoint_path)
    )
    t_total = time.time() - t_start
    trainer.logger.log_metrics({"trainer/total_time": t_total})
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    app()
