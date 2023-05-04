import os

import torch
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything

import datamodules
import models

PATH_DATASETS = "./data"
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = os.cpu_count() - 1


def train(seed=7, *, use_wandb=False):
    seed_everything(seed)
    # set model and data
    datamodule = datamodules.CIFAR10(
        data_dir=PATH_DATASETS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    model = models.Resnet(lr=0.05)
    # set trainer
    # TODO: wandb.init(save_code=True)
    logger = [loggers.CSVLogger(save_dir="logs/")]
    if use_wandb:
        project_name = f"{model.__name__.lower()}_{datamodule.__name__.lower()}"
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
    # TODO: wandb.finish()?


if __name__ == "__main__":
    main()
