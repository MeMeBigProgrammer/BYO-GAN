import math
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


def display_image(
    images,
    num_display=4,
    save_to_disk=False,
    save_dir="./output",
    filename="figure",
    title="Images",
):
    if images.dim() == 3:  # single image
        plt.imshow(images.permute(1, 2, 0))

    else:  # multiple images
        nrow = int(math.sqrt(num_display))

        image_grid = utils.make_grid(images.detach().cpu()[:num_display], nrow=nrow)

        plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    plt.title(title)

    if save_to_disk:
        plt.savefig("{0}/{1}.png".format(save_dir, filename))
    else:
        plt.show()


def get_truncated_noise(n_samples, z_dim, truncation):
    truncated_noise = truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim))
    return torch.Tensor(truncated_noise)
