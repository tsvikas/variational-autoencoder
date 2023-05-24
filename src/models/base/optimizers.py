import torch


class SGD:
    optimizer_cls = torch.optim.SGD
    optimizer_hparams = ["lr", "momentum", "weight_decay"]


class Adam:
    optimizer_cls = torch.optim.Adam
    optimizer_hparams = ["lr"]
