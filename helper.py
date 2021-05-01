import torch
import matplotlib.pyplot as plt
from torchvision import utils
from scipy.stats import truncnorm
from math import sqrt
import os

"""
Displays/Saves a single image or multiple images in a grid using pyplot.

Function output image is limited in resolution and therefore quality.
"""


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
        nrow = int(sqrt(num_display))

        image_grid = utils.make_grid(images.detach().cpu()[:num_display], nrow=nrow)

        plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    plt.title(title)

    if save_to_disk:
        plt.savefig("{0}/{1}.png".format(save_dir, filename))
    else:
        plt.show()


"""
Creates new subfolder in root directory.
Saves each image seperately in full resolution.
"""


def save_image_samples(
    images, fold_name, root_dir="./output/samples", image_prefix="sample", max_images=16
):
    for index, image_tensor in enumerate(images):
        folder_path = root_dir + f"/{fold_name}"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if index < max_images:
            image_name = folder_path + f"/{image_prefix}-{index + 1}.png"
            utils.save_image(image_tensor, image_name)


def get_truncated_noise(n_samples, z_dim, trunc):
    noise = (
        torch.as_tensor(
            truncnorm.rvs(-trunc, trunc, size=(n_samples, z_dim)),
            dtype=torch.float,
        )
        .cuda()
        .requires_grad_()
    )
    return noise


def set_requires_grad(model, requires_grad: bool):
    for p in model.parameters():
        p.requires_grad = requires_grad
