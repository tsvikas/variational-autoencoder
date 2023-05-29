# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from IPython.core.display_functions import display
from ipywidgets import interact
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

from datamodules import ImagesDataModule
from models import FullyConnectedAutoEncoder

# %%
ckpt_dir = (
    Path("/tmp/logs")
    / "fullyconnectedautoencodersgd-fashionmnist"
    / "fullyconnectedautoencodersgd-fashionmnist"
)
for p in ckpt_dir.parents[::-1] + (ckpt_dir,):
    if not p.exists():
        raise ValueError(f"{p} not exists")


def get_last_fn(p: Path):
    last_fn = max(p.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime)
    return (
        datetime.fromtimestamp(last_fn.stat().st_mtime).isoformat(" "),  # noqa: DTZ006
        last_fn.relative_to(p.parent).as_posix(),
    )


def sort_dict(d: dict):
    d = dict(d)
    return {k: d[k] for k in sorted(d)}


all_ckpts = sort_dict(get_last_fn(subdir) for subdir in ckpt_dir.glob("*"))
display(all_ckpts)

# %%
# torch.load(ckpt_dir/list(all_ckpts.values())[-1])['hyper_parameters']

# %%
model = FullyConnectedAutoEncoder.load_latest_checkpoint(ckpt_dir)
model.eval()
print(model.hparams)
print(model)

# %%
x_rand = torch.rand(1, 1, 28, 28)
image = ImagesDataModule("FashionMNIST", 1, 10).dataset()[0][0]

x_real = ToTensor()(image).unsqueeze(0)
print(x_real.shape)


# %%
def show_tensors(imgs: list[torch.Tensor]):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axss = plt.subplots(ncols=len(imgs), squeeze=False)
    axs = axss[0]
    for i, img in enumerate(imgs):
        img_clipped = img.detach().clip(0, 1)
        img_pil = to_pil_image(img_clipped)
        axs[i].imshow(img_pil, cmap="gray", vmin=0, vmax=255)
        axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


for x in [x_rand, x_real]:
    show_tensors([x[0], model(x.cuda())[0]])

# %%
n_latent = 8

lims = (-2, 2, 0.01)
all_lims = {f"x{i:02}": lims for i in range(n_latent)}


def show_from_latent(**inputs):
    data = torch.tensor(list(inputs.values()))
    data = data.view(1, -1).cuda()
    result = model.decoder(data)[0]
    show_tensors(result)
    plt.show()


interact(show_from_latent, **all_lims)

# %%
