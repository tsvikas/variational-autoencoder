import dataclasses
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torchmetrics.functional.classification import multiclass_accuracy


class HasXHat:
    x_hat: torch.Tensor


@dataclasses.dataclass
class AutoEncoderOutput:
    x_hat: torch.Tensor


@dataclasses.dataclass
class VAEOutput(AutoEncoderOutput):
    x_hat: torch.Tensor
    mu: torch.Tensor
    log_var_2: torch.Tensor


class SimpleLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.total_examples = 0

    @classmethod
    def load_latest_checkpoint(cls, base_dir: Path, **kwargs):
        all_checkpoints = sorted(
            base_dir.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime
        )
        if not all_checkpoints:
            raise RuntimeError("no checkpoints found")
        ckpt_fn_latest = all_checkpoints[-1]
        print(f"loading from {ckpt_fn_latest}")  # noqa: T201
        return cls.load_from_checkpoint(ckpt_fn_latest, **kwargs)

    def step(self, batch, batch_idx, stage: str, *, evaluate=False):
        """
        run a general training/validation/test step.
        return the loss
        evaluate is True if stage is validation or test
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "training")
        self.total_examples += len(batch[0])
        self.log("trainer/total_examples", float(self.total_examples))
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("trainer/total_examples", float(self.total_examples))
        self.step(batch, batch_idx, "validation", evaluate=True)

    def test_step(self, batch, batch_idx):
        self.log("trainer/total_examples", float(self.total_examples))
        self.step(batch, batch_idx, "test", evaluate=True)


class LightningModuleWithOptimizer(SimpleLightningModule):
    def __init__(self, optimizer_cls=None, optimizer_kwargs=None):
        super().__init__()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}

    def configure_optimizers(self):
        if self.optimizer_cls is None:
            raise ValueError("must define an optimizer_cls")
        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        return optimizer


class LightningModuleWithScheduler(LightningModuleWithOptimizer):
    def __init__(
        self,
        *,
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        scheduler_interval=None,
        scheduler_frequency=None,
        scheduler_add_total_steps=None,
        scheduler_monitor=None,
    ):
        super().__init__(optimizer_cls=optimizer_cls, optimizer_kwargs=optimizer_kwargs)
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.scheduler_add_total_steps = scheduler_add_total_steps
        self.scheduler_monitor = scheduler_monitor

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        if self.scheduler_cls is None:
            return optimizer
        scheduler_interval_options = ["epoch", "step"]
        if self.scheduler_interval not in scheduler_interval_options:
            raise ValueError(f"scheduler_interval not in {scheduler_interval_options}")
        created_scheduler_kwargs = {}
        if self.scheduler_add_total_steps:
            total_steps = self.trainer.estimated_stepping_batches
            created_scheduler_kwargs["total_steps"] = total_steps
        lr_scheduler = {
            "scheduler": self.scheduler_cls(
                optimizer, **self.scheduler_kwargs, **created_scheduler_kwargs
            ),
            "interval": self.scheduler_interval,
        }
        if self.scheduler_monitor is not None:
            lr_scheduler["monitor"] = self.scheduler_monitor
        if self.scheduler_frequency is not None:
            lr_scheduler["frequency"] = self.scheduler_frequency
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class NLLClassifier(LightningModuleWithScheduler):
    """
    classifier that returns the negative-log-likelihood of each class.
    uses nll_loss
    """

    def __init__(
        self,
        num_classes: int,
        *,
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        scheduler_interval="epoch",
        scheduler_frequency=None,
        scheduler_add_total_steps=False,
        scheduler_monitor=None,
    ):
        super().__init__(
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_interval=scheduler_interval,
            scheduler_frequency=scheduler_frequency,
            scheduler_add_total_steps=scheduler_add_total_steps,
            scheduler_monitor=scheduler_monitor,
        )
        self.num_classes = num_classes

    def step(self, batch, batch_idx, stage: str, *, evaluate=False):
        x, target = batch
        nll = self(x)
        loss = F.nll_loss(nll, target)
        self.log(f"loss/{stage}", loss, prog_bar=evaluate)
        if evaluate:
            acc = multiclass_accuracy(nll, target, num_classes=self.num_classes)
            self.log(f"accuracy/{stage}", acc, prog_bar=True)
        return loss


class ImageClassifier(NLLClassifier):
    def __init__(
        self,
        image_size: int,
        num_channels: int,
        num_classes: int,
        *,
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        scheduler_interval="epoch",
        scheduler_frequency=None,
        scheduler_add_total_steps=False,
        scheduler_monitor=None,
    ):
        super().__init__(
            num_classes=num_classes,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_interval=scheduler_interval,
            scheduler_frequency=scheduler_frequency,
            scheduler_add_total_steps=scheduler_add_total_steps,
            scheduler_monitor=scheduler_monitor,
        )
        sample_batch_size = 32
        self.image_size = image_size or 96
        self.num_channels = num_channels
        self.example_input_array = torch.empty(
            sample_batch_size, num_channels, self.image_size, self.image_size
        )


class AutoEncoder(LightningModuleWithScheduler):
    """
    encoder that tries to return itself.
    uses mse_loss
    """

    n_images_to_save = 8

    def __init__(
        self,
        sampler_lim: float = 3.0,
        sampler_n: int = 30,
        sampler_downsample_factor: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sampler_n = sampler_n
        self.sampler_downsample_factor = sampler_downsample_factor
        self.register_buffer(
            "sampler_x", torch.linspace(-sampler_lim, sampler_lim, sampler_n)
        )
        self.register_buffer(
            "sampler_y", torch.linspace(-sampler_lim, sampler_lim, sampler_n)
        )
        self.register_buffer(
            "sampler_xy", torch.cartesian_prod(self.sampler_x, self.sampler_y)
        )

    def sample_latent(self):
        if self.latent_dim != 2:
            return None
        with torch.inference_mode():
            outs = self.decoder(self.sampler_xy)
            out = rearrange(
                outs, "(i j) c h w -> c (i h) (j w)", i=self.sampler_n, j=self.sampler_n
            )
            out = torch.nn.functional.avg_pool2d(
                out, kernel_size=self.sampler_downsample_factor
            )
        return out

    def loss_function(self, batch, out: HasXHat):
        x, target = batch
        assert out.x_hat.shape == x.shape
        loss = F.mse_loss(out.x_hat, target)
        return loss

    def step(self, batch, batch_idx, stage: str, *, evaluate=False):
        x, target = batch
        assert x.shape == target.shape
        out: AutoEncoderOutput = self(x)
        loss = self.loss_function(batch, out)

        if isinstance(loss, dict):
            assert "loss" in loss, "Primary loss must be present"
            for k, v in loss.items():
                self.log(f"{k}/{stage}", v, prog_bar=evaluate)
            loss = loss["loss"]
        else:
            self.log(f"loss/{stage}", loss, prog_bar=evaluate)

        if stage == "validation" and self.logger:
            if batch_idx == 0:
                self.logger.log_image("image/src", list(x[: self.n_images_to_save]))
            if self.global_step == 0 and batch_idx == 0:
                self.logger.log_image(
                    "image/target", list(target[: self.n_images_to_save])
                )
            if batch_idx == 0:
                self.logger.log_image(
                    "image/pred", list(out.x_hat[: self.n_images_to_save])
                )

        if stage == "validation" and self.logger and batch_idx == 0:
            sampled_images = self.sample_latent()
            if sampled_images is not None:
                self.logger.log_image("image/sampled", [sampled_images])

        return loss


class VAE(AutoEncoder):
    def __init__(self, kl_weight: float = 0.005, *args, **kwargs):
        """
        Magic constant from https://github.com/AntixK/PyTorch-VAE/
        """
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight

    def loss_function(self, batch, out: VAEOutput) -> dict[str, torch.Tensor]:
        x, target = batch
        assert out.x_hat.shape == x.shape
        reconstruction_loss = F.mse_loss(out.x_hat, target)
        log_var_2 = out.log_var_2
        mu = out.mu

        kl_loss = (
            0.5
            * (-log_var_2 + log_var_2.exp() + mu**2 - 1).sum(dim=1).mean(dim=0)
            * self.kl_weight
        )
        loss = reconstruction_loss + kl_loss
        return dict(
            loss=loss,
            reconstruction_loss=reconstruction_loss.detach(),
            kl_loss=kl_loss.detach(),
        )


class ImageAutoEncoder(AutoEncoder):
    def __init__(
        self,
        image_size: int,
        num_channels: int,
        *,
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        scheduler_interval="epoch",
        scheduler_frequency=None,
        scheduler_add_total_steps=False,
        scheduler_monitor=None,
    ):
        super().__init__(
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_interval=scheduler_interval,
            scheduler_frequency=scheduler_frequency,
            scheduler_add_total_steps=scheduler_add_total_steps,
            scheduler_monitor=scheduler_monitor,
        )
        sample_batch_size = 32
        self.image_size = image_size or 96
        self.num_channels = num_channels
        self.example_input_array = torch.empty(
            sample_batch_size, num_channels, self.image_size, self.image_size
        )


class ImageVAE(VAE):
    def __init__(
        self,
        image_size: int,
        num_channels: int,
        kl_weight: float,
        *,
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        scheduler_interval="epoch",
        scheduler_frequency=None,
        scheduler_add_total_steps=False,
        scheduler_monitor=None,
    ):
        super().__init__(
            kl_weight=kl_weight,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_interval=scheduler_interval,
            scheduler_frequency=scheduler_frequency,
            scheduler_add_total_steps=scheduler_add_total_steps,
            scheduler_monitor=scheduler_monitor,
        )
        sample_batch_size = 32
        self.image_size = image_size or 96
        self.num_channels = num_channels
        self.example_input_array = torch.empty(
            sample_batch_size, num_channels, self.image_size, self.image_size
        )
