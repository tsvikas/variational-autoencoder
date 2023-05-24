import torch


class OneCycleLR:
    scheduler_cls = torch.optim.lr_scheduler.OneCycleLR
    scheduler_hparams = ["max_lr"]
    scheduler_interval = "step"
    add_total_steps = True


class ExponentialLR:
    scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    scheduler_interval = "epoch"
    scheduler_hparams = ["gamma"]
    add_total_steps = False
