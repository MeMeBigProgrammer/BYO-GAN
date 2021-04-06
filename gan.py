# Formatted using 'autopep8' and 'black' VS Code Extentions

import sys
import torch
import torchvision
import gc
import math
from torch import nn
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.stats import truncnorm
from datetime import datetime


"""
Assumptions:

1. The progression begins at 4x4, and goes until 512x512, upsampling by factors of 2 (4x4 -> 8x8).
2. Noise is ALWAYS 512.

TODOs:
- Specific runtime initializations 
- Allow for PixelWise normalization after each GAN Block 
"""


class InjectSecondaryNoise(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones((1, num_channels, 1, 1)))

    def forward(self, conv_output, noise):
        # noise_shape = (
        #     conv_output.shape[0],
        #     1,
        #     conv_output.shape[2],
        #     conv_output.shape[3],
        # )

        return conv_output + (self.weights * noise)


class AdaINBlock(nn.Module):
    def __init__(self, num_channels, style_length=512):
        super().__init__()

        self.num_channels = num_channels

        self.instance_norm = nn.InstanceNorm2d(num_channels)
        self.lin = nn.Linear(style_length, 2 * num_channels)

    def forward(self, image, noise):
        y_style = self.lin(noise).view(-1, 2, self.num_channels, 1, 1)
        inst_norm = self.instance_norm(image)
        return (inst_norm * y_style[:, 0]) + y_style[:, 1]


class StyleGanBlock(nn.Module):
    def __init__(self, in_chan, out_chan, is_initial=False, does_upsample=True):
        super().__init__()

    def forward(self):
        print()


class MappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        print()


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        print()


class Generator(nn.Module):
    def __init__(self, channel_progression: list):
        super().__init__()

        """
        We need: Channel progression
        list of Conv Blocks
        list of RGB conversions

        """
        self.gen_blocks = nn.ModuleList([])

    def forward(self):
        print()


if __name__ == "__main__":
    if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
        print(torch.cuda.get_device_name(0))