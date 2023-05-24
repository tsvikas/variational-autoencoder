# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from ipywidgets import interact
from models import FullyConnectedAutoEncoderSGD
from torchvision.transforms.functional import to_pil_image

# %%
ckpt_dir = (
    Path("/tmp/logs")
    / "fullyconnectedautoencoder-fashionmnist"
    / "fullyconnectedautoencoder-fashionmnist"
)

# %%
model = FullyConnectedAutoEncoderSGD.load_latest_checkpoint(ckpt_dir, num_channels=1)
model.eval()
print(model.hparams)
print(model)

# %%
x = torch.rand(1, 1, 28, 28).cuda()


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


show_tensors([x[0], model(x)[0]])


# %%
def show_from_lateral(x1: float, x2: float, x3: float, x4: float):
    data = torch.tensor([x1, x2, x3, x4])
    data = data.view(1, -1).cuda()
    result = model.decoder(data)[0]
    show_tensors(result)
    plt.show()


lims = (-2, 2, 0.01)
interact(show_from_lateral, x1=lims, x2=lims, x3=lims, x4=lims)

# %%
