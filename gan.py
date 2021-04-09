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
- Specific runtime initializations* 
- Allow for PixelWise normalization after each GAN Block*
- Equalized Learning rate* 
- Dynamically create channel progression
- Device specification
- multiple latent noise inputs
"""


class InjectSecondaryNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones((1, channels, 1, 1)))

    def forward(self, conv_output, noise=None):
        if noise = None:
            noise_shape = (
                conv_output.shape[0],
                1,
                conv_output.shape[2],
                conv_output.shape[3],
            )
            noise = torch.randn(noise_shape)


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
        self.inject_noise = InjectSecondaryNoise(out_chan)
        self.adain = AdaINBlock(out_chan)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, style, noise=None):
        out = self.conv(x)
        out = self.inject_noise(out, noise=noise)
        out = self.adain(out, style)
        return self.activation(out)


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

    def forward(self, x, style, batch_size, noise=None):
        if not self.is_initial and x is None:
            raise ValueError(
                "If this is not the starting block, you will need input for the convolution."
            )

        if self.does_upsample:
            x = self.upsample(x)

        if self.is_initial:
            out = self.conv_1.repeat(batch_size, 1, 1, 1)
        else:
            out = self.conv_1(x, style, noise)

        out = self.conv_2(out, style, noise)


class MappingLayers(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.layers = nn.Sequential(
            self.generate_mapping_block(channels),
            self.generate_mapping_block(channels),
            self.generate_mapping_block(channels),
            self.generate_mapping_block(channels),
            self.generate_mapping_block(channels),
            self.generate_mapping_block(channels),
            self.generate_mapping_block(channels),
            self.generate_mapping_block(channels),
        )

    def generate_mapping_block(self, channels: int):
        return nn.Sequential(nn.Linear(channels, channels), nn.LeakyReLU(negative_slope=0.2))

    def forward(self, input):
        return self.layers(input)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.to_w_noise = MappingLayers()

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

        self.to_rgbs = nn.ModuleList(
            nn.Conv2d(512, 3, kernel_size=1),
            nn.Conv2d(512, 3, kernel_size=1),
            nn.Conv2d(512, 3, kernel_size=1),
            nn.Conv2d(256, 3, kernel_size=1),
            nn.Conv2d(128, 3, kernel_size=1),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Conv2d(16, 3, kernel_size=1),
        )

    def forward(self, z_noise, noise=None, steps=1, alpha=None):
        style = self.to_w_noise(z_noise)

        out = None

        for index, (to_rgb, gen_block) in enumerate(zip(self.to_rgbs, self.gen_blocks)):

            previous = out

            out = gen_block.forward(out, style, noise)

            if (index + 1) >= steps: # final step
                if alpha is not None and index > 0: # mix final image and return
                    
                    # clamp alpha to 0 -> 1
                    alpha = min(1.0, max(0.0, alpha))

                    small_image_upsample = to_rgbs[index - 1](previous)
                    large_image = to_rgb(out)

                    return torch.lerp(small_image_upsample, large_image, alpha)
                else: # No fad in.
                    return to_rgb(out)

class CriticBlock(nn.Module):
    def __init__(self, in_chan, out_chan, is_final_layer=False, downsample=True):
        super().__init__()

        # 2 Conv layers


    def forward(self):
        print()

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        print()

def train():
    print("Hello There!")
    gen = Generator()

if __name__ == "__main__":
    if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
        print(torch.cuda.get_device_name(0))
    
    train()