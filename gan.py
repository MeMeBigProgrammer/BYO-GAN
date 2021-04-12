# Formatted using 'autopep8' and 'black' VS Code Extentions

import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


"""
Assumptions:

1. The progression begins at 4x4, and goes until 512x512, upsampling by factors of 2 (4x4 -> 8x8).
2. Noise is ALWAYS 512.

TODOs:
- Exponential Moving Average*
- Dynamically create channel progression
- Device specification
- multiple latent noise inputs

CHECK:
- tensor.detach()
"""


class EqualizedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        torch.nn.init.normal_(self.weight)
        torch.nn.init.zeros_(self.bias)

        self.scale = sqrt(
            2 / (self.kernel_size[0] * self.kernel_size[1] * self.in_channels)
        )

    def forward(self, x):
        return torch.conv2d(
            input=x,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class EqualizedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        torch.nn.init.normal_(self.weight)
        torch.nn.init.zeros_(self.bias)

        self.scale = sqrt(2 / self.in_features)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight * self.scale, self.bias)


a = EqualizedLinear(1, 1)


class InjectSecondaryNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones((1, channels, 1, 1)))

    def forward(self, conv_output, noise=None):
        if noise is None:
            noise_shape = (
                conv_output.shape[0],
                1,
                conv_output.shape[2],
                conv_output.shape[3],
            )
            noise = torch.randn(noise_shape, device=conv_output.device)

        return conv_output + (self.weights * noise)


class AdaINBlock(nn.Module):
    def __init__(self, channels, style_length=512):
        super().__init__()

        self.channels = channels

        self.instance_norm = nn.InstanceNorm2d(channels)
        self.lin = EqualizedLinear(style_length, 2 * channels)

    def forward(self, image, noise):
        y_style = self.lin(noise).view(-1, 2, self.channels, 1, 1)
        inst_norm = self.instance_norm(image)
        return (inst_norm * y_style[:, 0]) + y_style[:, 1]


class StyleConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv = EqualizedConv2d(in_chan, out_chan, kernel_size=3, padding=1)
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
            raise ValueError("You cannot use the Starting Constant and Upsample.")

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
            raise ValueError("Expected x to not be None.")

        if self.does_upsample:
            x = self.upsample(x)

        if self.is_initial:
            out = self.conv_1.repeat(batch_size, 1, 1, 1)
        else:
            out = self.conv_1(x, style, noise)

        return self.conv_2(out, style, noise)


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
        return nn.Sequential(
            EqualizedLinear(channels, channels), nn.LeakyReLU(negative_slope=0.2)
        )

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
            [
                EqualizedConv2d(512, 3, kernel_size=1),
                EqualizedConv2d(512, 3, kernel_size=1),
                EqualizedConv2d(512, 3, kernel_size=1),
                EqualizedConv2d(256, 3, kernel_size=1),
                EqualizedConv2d(128, 3, kernel_size=1),
                EqualizedConv2d(64, 3, kernel_size=1),
                EqualizedConv2d(32, 3, kernel_size=1),
                EqualizedConv2d(16, 3, kernel_size=1),
            ]
        )

    def forward(self, z_noise, noise=None, steps=1, alpha=None):
        normalized_noise = z_noise / torch.sqrt(
            torch.mean(z_noise ** 2, dim=1, keepdim=True) + 1e-8
        )

        style = self.to_w_noise(normalized_noise)

        out = None

        for index, (to_rgb, gen_block) in enumerate(zip(self.to_rgbs, self.gen_blocks)):

            previous = out

            out = gen_block.forward(out, style, len(z_noise), noise=noise)

            if (index + 1) >= steps:  # final step
                if alpha is not None and index > 0:  # mix final image and return

                    # clamp alpha to 0 -> 1
                    alpha = min(1.0, max(0.0, alpha))

                    small_image_upsample = F.interpolate(
                        self.to_rgbs[index - 1](previous),
                        scale_factor=2,
                        mode="bilinear",
                    )
                    large_image = to_rgb(out)

                    return torch.lerp(small_image_upsample, large_image, alpha)
                else:  # No fad in.
                    return to_rgb(out)

    def get_loss(self, crit_fake_pred):
        return -crit_fake_pred.mean()


class CriticBlock(nn.Module):
    def __init__(self, in_chan, out_chan, is_final_layer=False):
        super().__init__()

        self.is_final_layer = is_final_layer

        if is_final_layer:
            self.conv_1 = nn.Sequential(
                MiniBatchStdDev(),
                EqualizedConv2d(in_chan + 1, out_chan, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
            )

            self.conv_2 = nn.Sequential(
                EqualizedConv2d(out_chan, out_chan, kernel_size=4),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                EqualizedLinear(out_chan, 1),
            )
        else:
            self.conv_1 = nn.Sequential(
                EqualizedConv2d(in_chan, out_chan, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
            )

            self.conv_2 = nn.Sequential(
                EqualizedConv2d(out_chan, out_chan, kernel_size=3, padding=1),
                nn.AvgPool2d(2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):

        (batch_size, channels, h, w) = x.shape

        if self.group_size % batch_size != 0:
            self.group_size = batch_size

        minibatch = x.reshape([self.group_size, -1, 1, channels, h, w])

        minibatch_means = x.mean(0, keepdim=True)

        minibatch_variance = ((minibatch - minibatch_means) ** 2).mean(0, keepdim=True)

        minibatch_std = (
            ((minibatch_variance + 1e-8) ** 0.5)
            .mean([3, 4, 5], keepdim=True)
            .squeeze(3)
        )

        minibatch_std = (
            minibatch_std.expand(self.group_size, -1, -1, h, w)
            .clone()
            .reshape(batch_size, 1, h, w)
        )

        return torch.cat([x, minibatch_std], dim=1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.ModuleList(
            [
                CriticBlock(16, 32),
                CriticBlock(32, 64),
                CriticBlock(64, 128),
                CriticBlock(128, 256),
                CriticBlock(256, 512),
                CriticBlock(512, 512),
                CriticBlock(512, 512),
                CriticBlock(512, 512, is_final_layer=True),
            ]
        )

        self.from_rgbs = nn.ModuleList(
            [
                self.gen_from_rgbs(16),
                self.gen_from_rgbs(32),
                self.gen_from_rgbs(64),
                self.gen_from_rgbs(128),
                self.gen_from_rgbs(256),
                self.gen_from_rgbs(512),
                self.gen_from_rgbs(512),
                self.gen_from_rgbs(512),
            ]
        )

    def forward(self, images, steps=1, alpha=None):
        out = None
        n_blocks = len(self.conv_blocks)
        start = n_blocks - steps

        for index, conv_block in enumerate(self.conv_blocks[start:]):
            if index == 0:
                out = self.from_rgbs[start](images)

            out = conv_block(out)

            if index == 0 and steps > 1 and alpha is not None:
                # clamp alpha to 0 -> 1
                alpha = min(1.0, max(0.0, alpha))
                simple_downsample = self.from_rgbs[start + 1](F.avg_pool2d(images, 2))

                out = torch.lerp(simple_downsample, out, alpha)

        return out

    def gen_from_rgbs(self, out_chan, image_chan=3):
        # You can add a leaky relu activation too!
        return nn.Sequential(EqualizedConv2d(image_chan, out_chan, kernel_size=1))

    def get_loss(
        self,
        crit_fake_pred,
        crit_real_pred,
        epsilon,
        real_images,
        fake_images,
        steps,
        alpha,
        c_lambda,
    ):

        # Create mixed images and calculate gradient.
        mixed_images = real_images * epsilon + (1 - epsilon) * fake_images
        mixed_image_scores = self.forward(mixed_images, steps=steps, alpha=alpha)

        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_image_scores,
            grad_outputs=torch.ones_like(mixed_image_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Create gradient penalty.

        grad_penalty = (
            (gradient.view(gradient.size(0), -1).norm(2, dim=1) - 1) ** 2
        ).mean()

        # Put it all together.

        diff = -(crit_real_pred.mean() - crit_fake_pred.mean())

        gp = c_lambda * grad_penalty

        return diff + gp
