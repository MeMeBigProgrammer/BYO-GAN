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
- Equalized Learning Rate
- Dynamically create channel progression
- Allow for PixelWise normalization after each GAN Block 
"""


class InjectSecondaryNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones((1, channels, 1, 1)))

    def forward(self, conv_output, noise):
        # noise_shape = (
        #     conv_output.shape[0],
        #     1,
        #     conv_output.shape[2],
        #     conv_output.shape[3],
        # )

        return conv_output + (self.weights * noise)


class AdaINBlock(nn.Module):
    def __init__(self, channels, style_length=512):
        super().__init__()

        self.channels = channels

        self.instance_norm = nn.InstanceNorm2d(channels)
        self.lin = nn.Linear(style_length, 2 * channels)

    def forward(self, image, noise):
        y_style = self.lin(noise).view(-1, 2, self.channels, 1, 1)
        inst_norm = self.instance_norm(image)
        return (inst_norm * y_style[:, 0]) + y_style[:, 1]


class StyleConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, kernel_size=3, padding=1)
        self.inject_style = InjectSecondaryNoise(out_chan)
        self.adain = AdaINBlock(out_chan)

    def forward(self, x, w_noise, style):
        out = self.conv(x)
        out = self.inject_style(out, style)
        return self.adain(out, w_noise)


class StyleGanBlock(nn.Module):
    def __init__(self, in_chan, out_chan, is_initial=False, does_upsample=True):
        super().__init__()

        if is_initial and does_upsample:
            raise ValueError(
                "You are trying to use the Starting Constant and Upsample????"
            )

        self.is_initial = is_initial

        self.does_upsample = does_upsample

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        if is_initial:
            self.conv_1 = nn.Parameter(torch.ones(1, in_chan, 4, 4))
        else:
            self.conv_1 = StyleConvBlock(in_chan, out_chan)

        self.conv_2 = StyleConvBlock(out_chan, out_chan)

    def forward(self, x, w_noise, style, batch_size):
        if not self.is_initial and x is None:
            raise ValueError(
                "If this is not the starting block, you will need input for the convolution."
            )

        if self.does_upsample:
            x = self.upsample(x)

        if self.is_initial:
            out = self.conv_1.repeat(batch_size, 1, 1, 1)
        else:
            out = self.conv_1(x, w_noise, style)

        out = self.conv_2(out, w_noise, style)


class MappingLayers(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=1024):
        super().__init__()
        self.layers = nn.Sequential(
            self.generate_mapping_block(in_channels, hidden_channels),
            self.generate_mapping_block(hidden_channels, hidden_channels),
            self.generate_mapping_block(hidden_channels, hidden_channels),
            self.generate_mapping_block(hidden_channels, hidden_channels),
            self.generate_mapping_block(hidden_channels, hidden_channels),
            self.generate_mapping_block(hidden_channels, hidden_channels),
            self.generate_mapping_block(hidden_channels, hidden_channels),
            nn.Linear(hidden_channels, in_channels),
        )

    def generate_mapping_block(self, in_chan: int, out_chan: int):
        return nn.Sequential(nn.Linear(in_chan, out_chan), nn.ReLU())

    def forward(self, input):
        return self.layers(input)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        print()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        """
        We need: Channel progression
        list of Conv Blocks
        list of RGB conversions

        """
        self.gen_blocks = nn.ModuleList(
            [
                StyleGanBlock(512, 512, is_initial=True, does_upsample=False),
                StyleGanBlock(512, 512),
                StyleGanBlock(512, 512),
                StyleGanBlock(512, 256),
                StyleGanBlock(256, 128),
                StyleGanBlock(128, 64),
                StyleGanBlock(64, 32),
                StyleGanBlock(32, 16),
            ]
        )

        self.to_rgb = nn.ModuleList(
            nn.Conv2d(512, 3, kernel_size=1),
            nn.Conv2d(512, 3, kernel_size=1),
            nn.Conv2d(512, 3, kernel_size=1),
            nn.Conv2d(256, 3, kernel_size=1),
            nn.Conv2d(128, 3, kernel_size=1),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Conv2d(16, 3, kernel_size=1),
        )

    def forward(self):
        print()


if __name__ == "__main__":
    if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
        print(torch.cuda.get_device_name(0))