import torch
import matplotlib.pyplot as plt
from torchvision import utils
from scipy.stats import truncnorm
from math import floor, modf, sqrt


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


def get_truncated_noise(n_samples, z_dim, truncation):
    # truncated_noise = truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim))
    return torch.randn((n_samples, z_dim), requires_grad=True).cuda()


def get_progression_step(im_count, im_milestone, max_steps=8):
    count = float(im_count / im_milestone)

    floored_count = floor(count)

    steps = (floored_count) / 2 + 1
    is_even = floored_count % 2 == 0

    if steps >= max_steps:
        return (None, max_steps)

    if steps <= 1:
        return (None, 1)
    elif is_even:
        return (None, int(steps))
    else:
        return (round(modf(count)[0], 5), int(steps + 1))


def set_requires_grad(model, requires_grad: bool):
    for p in model.parameters():
        p.requires_grad = requires_grad
