# Formatted using 'autopep8' and 'black' VS Code Extentions

import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


class EqualizedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.data.normal_()
        self.bias.data.zero_()

        fan_in = self.weight.data.size()[1] * self.weight.data[0][0].numel()
        self.equalized_coefficient = sqrt(2 / fan_in)

    def forward(self, x):
        return F.linear(x, self.weight * self.equalized_coefficient, self.bias)


class EqualizedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.data.normal_()
        self.bias.data.zero_()

        fan_in = self.weight.data.size()[1] * self.weight.data[0][0].numel()
        self.equalized_coefficient = sqrt(2 / fan_in)

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.equalized_coefficient,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class InjectSecondaryNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros((1, channels, 1, 1)))

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
    def __init__(self, in_channel, style_dim=512):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualizedLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class StyleConvBlock(nn.Module):  # refactor so constant sits here.
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv = EqualizedConv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.inject_noise = InjectSecondaryNoise(out_chan)
        self.activation = nn.LeakyReLU(0.2)
        self.adain = AdaINBlock(out_chan)

    def forward(self, x, style, noise=None):
        out = self.conv(x)
        out = self.inject_noise(out, noise=noise)
        out = self.activation(out)
        return self.adain(out, style)


class StyleGanBlock(nn.Module):
    def __init__(self, in_chan, out_chan, is_initial=False, does_upsample=True):
        super().__init__()

        if is_initial and does_upsample:
            raise ValueError("You cannot use the Starting Constant and Upsample.")

        self.is_initial = is_initial

        self.does_upsample = does_upsample

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        if is_initial:
            self.constant = nn.Parameter(torch.randn(1, in_chan, 4, 4))
            self.inject_noise = InjectSecondaryNoise(out_chan)
            self.activation = nn.LeakyReLU(0.2)
            self.adain = AdaINBlock(out_chan)
        else:
            self.conv_1 = StyleConvBlock(in_chan, out_chan)

        self.conv_2 = StyleConvBlock(out_chan, out_chan)

    def forward(self, x, style, batch_size, noise=None):
        if not self.is_initial and x is None:
            raise ValueError("Expected x to not be None.")

        if self.does_upsample:
            x = self.upsample(x)

        if self.is_initial:
            out = self.constant.repeat(batch_size, 1, 1, 1)
            out = self.inject_noise(out, noise=noise)
            out = self.activation(out)
            out = self.adain(out, style)
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
        return nn.Sequential(EqualizedLinear(channels, channels), nn.LeakyReLU(0.2))

    def forward(self, input):
        return self.layers(input)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.to_w_noise = nn.Sequential(MappingLayers())

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

        style = self.to_w_noise(z_noise)

        out = None

        noise = []

        for i in range(steps):
            size = 4 * 2 ** i
            noise.append(
                torch.randn(len(z_noise), 1, size, size, device=z_noise.device)
            )

        out = noise[0]

        for index, (to_rgb, gen_block) in enumerate(zip(self.to_rgbs, self.gen_blocks)):

            previous = out

            out = gen_block.forward(out, style, len(z_noise), noise=noise[index])

            if (index + 1) >= steps:  # final step
                if alpha is not None and index > 0:  # mix final image and return.

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

    def get_wgan_loss(self, crit_fake_pred):
        return -crit_fake_pred.mean()

    def get_r1_loss(self, crit_fake_pred):
        return F.softplus(-crit_fake_pred).mean()


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
                EqualizedLinear(out_chan, out_chan),
                nn.LeakyReLU(0.2),
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

        if batch_size % self.group_size != 0:
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
        # You can add a leaky relu activation.
        return nn.Sequential(
            EqualizedConv2d(image_chan, out_chan, kernel_size=1), nn.LeakyReLU(0.2)
        )

    def get_wgan_loss(
        self,
        crit_fake_pred,
        crit_real_pred,
        real_im,
        steps,
        alpha,
        c_lambda=1,
    ):

        epsilon = torch.rand(
            real_im.shape[0], 1, 1, 1, device=self.device, requires_grad=True
        )

        # Create mixed images and calculate gradient.
        mixed_images = real_im * epsilon + (1 - epsilon) * fake_im
        mixed_image_scores = self.forward(mixed_images, steps=steps, alpha=alpha)

        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_image_scores,
            grad_outputs=torch.ones_like(mixed_image_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Create gradient penalty.

        gp = ((gradient.view(gradient.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

        wgan_loss = -crit_real_pred.mean() + crit_fake_pred.mean() + (c_lambda * gp)

        wgan_loss.backward()

        return wgan_loss

    def get_r1_loss(
        self, crit_fake_pred, crit_real_pred, real_im, fake_im, steps, alpha, c_lambda=1
    ):
        real_predict = F.softplus(-crit_real_pred).mean()

        grad_real = torch.autograd.grad(
            outputs=crit_real_pred.sum(), inputs=real_im, create_graph=True
        )[0]
        grad_penalty = (
            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()
        grad_penalty = c_lambda / 2 * grad_penalty

        fake_predict = F.softplus(crit_fake_pred).mean()

        r1_loss = real_predict + fake_predict + grad_penalty

        r1_loss.backward()

        return r1_loss
