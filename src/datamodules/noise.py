import torch


class GaussianNoise(torch.nn.Module):
    def __init__(self, std=1.0):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor):
        noise = torch.randn_like(x) * self.std
        return x + noise


class SaltPepperNoise(torch.nn.Module):
    def __init__(self, p_salt=0.0, p_pepper=0.0):
        super().__init__()
        self.p_salt = p_salt
        self.p_pepper = p_pepper

    def forward(self, x: torch.Tensor):
        noise_mask = torch.rand_like(x)
        return x.where(noise_mask >= self.p_salt + self.p_pepper, 1).where(
            noise_mask >= self.p_pepper, -1
        )
