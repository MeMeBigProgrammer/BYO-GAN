import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
from scipy.stats import truncnorm
import gc
from datetime import datetime


def display_image(images, num_display=4, save_to_disk=False, save_dir='./output', filename="figure", title="Images"):
    if images.dim() == 3:  # single image
        plt.imshow(images.permute(1, 2, 0))
    else:  # multiple images, show first {num_display} in grid
        image_grid = utils.make_grid(images.detach().cpu(
        )[:num_display], nrow=int(math.sqrt(num_display)))
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    plt.title(title)

    if save_to_disk:
        plt.savefig('{0}/{1}.png'.format(save_dir, filename))
    else:
        plt.show()


def get_truncated_noise(n_samples, z_dim, truncation):
    truncated_noise = truncnorm.rvs(-truncation,
                                    truncation, size=(n_samples, z_dim))
    return torch.Tensor(truncated_noise)


def get_growth_position(image_progression, critic_count, im_milestone):

    count = float(critic_count / (im_milestone + 1))

    # The generator should still be outputting base size.
    if count < 1.0:
        return image_progression[0], 1.0

    # Number of gan upsizes added, 0 means its on the starting size.
    advances = math.floor(math.floor(count) / 2)

    if 2 + advances > len(image_progression):
        return image_progression[-1], 1.0

    return image_progression[1 + advances], (count % 1)


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
            nn.Linear(hidden_channels, in_channels)
        )

    def generate_mapping_block(self, in_chan: int, out_chan: int):
        return nn.Sequential(
            nn.Linear(in_chan, out_chan),
            nn.ReLU()
        )

    def forward(self, input):
        return self.layers(input)


class InjectSecondaryNoise(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weights = nn.Parameter(
            torch.ones((1, num_channels, 1, 1))
        )

    def forward(self, conv_output):
        noise_shape = (
            conv_output.shape[0], 1, conv_output.shape[2], conv_output.shape[3])

        noise = torch.randn(noise_shape).to('cuda')

        return conv_output + (self.weights * noise)


class AdaINBlock(nn.Module):
    def __init__(self, noise_length, num_channels):
        super().__init__()

        self.num_channels = num_channels

        self.instance_norm = nn.InstanceNorm2d(num_channels)
        self.lin = nn.Linear(noise_length, 2 * num_channels)

        # Initialization
        self.lin.weight.data.fill_(1)
        self.lin.bias.data.fill_(0)

    def forward(self, image, noise):
        y_style = self.lin(noise).view(-1, 2, self.num_channels, 1, 1)
        inst_norm = self.instance_norm(image)
        return inst_norm * y_style[:, 0] + y_style[:, 1]


class StyleGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_color_channels=3, noise_dim=512, im_out=(8, 8), im_in=(4, 4)):
        super().__init__()

        self.simple_upsample = nn.Upsample(im_out, mode='bilinear')

        self.conv_1 = nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1)
        self.inject_noise_1 = InjectSecondaryNoise(out_channels)
        self.AdaIN_1 = AdaINBlock(noise_dim, out_channels)
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1)
        self.inject_noise_2 = InjectSecondaryNoise(out_channels)
        self.AdaIN_2 = AdaINBlock(noise_dim, out_channels)
        self.activation_2 = nn.LeakyReLU(negative_slope=0.2)

        # These layers convert to 3 channels.

        # convert data after convolutions to three color channels
        self.large_block_to_image = nn.Conv2d(
            out_channels, image_color_channels, kernel_size=1)
        # convert small data to three color channels
        self.small_block_to_image = nn.Conv2d(
            in_channels, image_color_channels, kernel_size=1)

    def forward(self, x, noise, alpha=None, do_upsample=True):

        # progressive growth with alpha. input from last block -> convert straight to image -> upsample -> mix
        # On the other end, input -> convolutions -> convert to image -> mix

        upsampled_x = x
        if do_upsample:
            upsampled_x = self.simple_upsample(x)

        out = self.conv_1(upsampled_x)
        out = self.inject_noise_1(out)
        out = self.AdaIN_1(out, noise)
        out = self.activation_1(out)

        out = self.conv_2(out)
        out = self.inject_noise_2(out)
        out = self.AdaIN_2(out, noise)
        out = self.activation_2(out)

        if alpha == None:  # assuming that this is not the last block
            return out
        elif alpha >= 0:
            small_image = self.small_block_to_image(x)
            small_image_upsampled = self.simple_upsample(small_image)

            conv_image = self.large_block_to_image(out)

            return torch.lerp(small_image_upsampled, conv_image, alpha)
        else:
            raise ValueError('alpha not valid!')


class StyleGAN(nn.Module):
    def __init__(self, image_progression, z_size=512, image_channels=3):
        super().__init__()

        # Save parameters.
        self.z_size = z_size

        self.image_channels = image_channels

        self.image_progression = image_progression

        self.num_blocks = len(image_progression)

        # Z -> W
        self.noise_mapping = MappingLayers(z_size)

        # Synthesis network: starting constant, init to 1
        self.starting_constant = nn.Parameter(
            torch.ones((1, z_size, image_progression[0], image_progression[0]))
        )

        self.gan_blocks = nn.ModuleList([])

        for index, im_size in enumerate(image_progression):
            prev_size = (im_size, im_size)
            out_size = (im_size, im_size)
            if index != 0:
                prev_size = (
                    image_progression[index - 1], image_progression[index - 1])

            self.gan_blocks.append(
                StyleGANBlock(512, 512, image_color_channels=image_channels,
                              im_out=out_size, im_in=prev_size)
            )

    def forward(self, z_noise, critic_count, im_milestone):
        # calculate alphas
        output_size, alpha = get_growth_position(
            self.image_progression, critic_count, im_milestone)

        # forward pass
        noise = self.noise_mapping(z_noise)

        output = self.starting_constant

        for i in range(self.num_blocks):
            if image_progression[i] == output_size:
                return self.gan_blocks[i].forward(output, noise, alpha=alpha)
            else:
                output = self.gan_blocks[i].forward(output, noise)

        return output


def get_generator_loss(crit_fake_pred):
    return -crit_fake_pred.mean()


class Critic(nn.Module):
    def __init__(self, image_progression, image_channels=3):
        super().__init__()

        self.image_progression = image_progression

        self.image_channels = image_channels

        # Critic Blocks

        self.critic_blocks = nn.ModuleList([])

        for index, im_size in enumerate(self.image_progression[::-1]):
            in_size = (im_size, im_size)

            self.critic_blocks.add_module(f'{in_size} intake Critic Block', CriticBlock(
                in_size, image_channels=image_channels, in_size=512, out_size=512))  # TODO have dynamic critic weights

        # Final Layers once you've just downsampled to 4x4

        self.final_layers = nn.Sequential(
            MiniBatchStdDev(),
            nn.Conv2d(513, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

        self.from_rgb = nn.Conv2d(image_channels, 512, kernel_size=1)

    def forward(self, x, critic_count, im_milestone):
        out_size, alpha = get_growth_position(
            self.image_progression, critic_count, im_milestone)

        out = self.from_rgb(x)

        for module in self.critic_blocks:
            if out_size == module.im_in_size:
                out = module.forward(out, alpha=alpha)
            elif out_size > module.im_in_size[0]:
                out = module.forward(out)

            if out.shape[2] == 4 and out.shape[3] == 4:
                return self.final_layers(out)

    def get_critic_loss(self, crit_fake_pred, crit_real_pred, epsilon, real_images, fake_images, discriminator_count, im_milestone, c_lambda):

        # Create mixed images and calculate gradient.
        mixed_images = real_images * epsilon + (1 - epsilon) * fake_images
        mixed_image_scores = self.forward(
            mixed_images, discriminator_count, im_milestone)

        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_image_scores,
            grad_outputs=torch.ones_like(mixed_image_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Create gradient penalty.

        gradient = gradient.view(len(gradient), -1)

        gradient_norm = gradient.norm(2, dim=1)

        penalty = ((gradient_norm - 1)**2).mean()

        # Put it all together.

        diff = -(crit_real_pred.mean() - crit_fake_pred.mean())

        gp = c_lambda * penalty

        return diff + gp


class CriticBlock(nn.Module):
    def __init__(self, im_in_size, image_channels=3, in_size=512, out_size=512):
        super().__init__()

        self.im_in_size = im_in_size

        self.from_rgb = nn.Conv2d(image_channels, in_size, 1)

        self.main_block = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.downsample = nn.AvgPool2d(2)

    def forward(self, x, alpha=None):
        if alpha is None:
            return self.downsample(self.main_block(x))
        elif alpha >= 0:

            conv_out = self.downsample(self.main_block(self.from_rgb(x)))

            simple_downsample = self.from_rgb(self.downsample(x))

            return torch.lerp(simple_downsample, conv_out, alpha)


class MiniBatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                'Tensor dim must be 4, got {} instead.'.format(x.dim()))

        return torch.cat((x, torch.std(x[:, None], dim=2)), 1)


class Debug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# IMPORTANT CONSTANTS
batch_size = 24  # Image batch size; Depends on VRAM available
# Progressive Growth block fade in constant; Each progression (fade-in/stabilization period) lasts X images
im_milestone = 200 * 1000
c_lambda = 10  # WGAN-GP Gradient Penalty coefficient
noise_size = 512
device = 'cuda'
beta_1 = 0  # For Adam optimizer
beta_2 = 0.99  # For Adam optimizer
learning_rate = 0.0001
critic_repeats = 1  # per Kerras et al
gen_weight_decay = 0.999

num_epochs = 500
display_step = 50
checkpoint_step = 1000

image_progression = [4, 8, 16, 32, 64, 128]

# Create a constant set of noise vectors to show same image progression.
show_noise = get_truncated_noise(4, 512, 0.75).to(device)

# LOADING DATA
transformation = transforms.Compose([
    transforms.Resize((image_progression[-1], image_progression[-1])),
    transforms.CenterCrop((image_progression[-1], image_progression[-1])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    transforms.ConvertImageDtype(float),
])

path_root = "/media/simba/407150955DB167D7/data"

glacier_images = datasets.ImageFolder(
    path_root + '/glaciers', transformation)
building_images = datasets.ImageFolder(
    path_root + '/buildings', transformation)
forest_images = datasets.ImageFolder(
    path_root + '/forest', transformation)
anime_images = datasets.ImageFolder(path_root + '/anime', transformation)

images = torch.utils.data.DataLoader(
    anime_images, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=3)

# Training Loop


def train():

    # Initialize Generator
    gen = StyleGAN(image_progression, z_size=noise_size).to(device)
    gen_opt = torch.optim.Adam([
        {'params': gen.noise_mapping.parameters(), 'lr': (learning_rate * 0.01)},
        {'params': gen.starting_constant},
        {'params': gen.gan_blocks.parameters()}], lr=learning_rate, betas=(
        beta_1, beta_2), weight_decay=gen_weight_decay)

    # Initialize Critic
    critic = Critic(image_progression).to(device)
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

    critic_image_count = 0
    step = 0
    g_loss_history = []
    c_loss_history = []

    avg_g_loss = 0
    avg_c_loss = 0

    for epoch in range(num_epochs):

        pbar = tqdm(images)

        for x, _ in pbar:
            current_batch_size = len(x)

            for i in range(critic_repeats):
                critic_opt.zero_grad()
                fake_noise = get_truncated_noise(
                    current_batch_size, noise_size, 0.75).to(device)

                fake_images = gen.forward(
                    fake_noise, critic_image_count, im_milestone)

                x = torch.nn.functional.interpolate(x, size=(
                    fake_images.shape[2], fake_images.shape[3]), mode='bilinear').to(device, dtype=torch.float)

                critic_fake_pred = critic.forward(
                    fake_images.detach(), critic_image_count, im_milestone)

                critic_real_pred = critic.forward(
                    x, critic_image_count, im_milestone)

                epsilon = torch.rand(
                    len(x), 1, 1, 1, device=device, requires_grad=True)

                loss = critic.get_critic_loss(critic_fake_pred, critic_real_pred,
                                              epsilon, x, fake_images, critic_image_count, im_milestone, c_lambda)

                loss.backward(retain_graph=True)

                critic_opt.step()

                critic_image_count += current_batch_size

                c_loss_history.append(loss.item())

            gen_opt.zero_grad()
            noise = get_truncated_noise(
                current_batch_size, noise_size, 0.75).to(device)
            fake_images = gen(
                noise, critic_image_count, im_milestone)

            critic_fake_pred = critic(
                fake_images, critic_image_count, im_milestone)

            gen_loss = get_generator_loss(critic_fake_pred)
            gen_loss.backward()

            gen_opt.step()

            g_loss_history.append(gen_loss.item())

            step += 1

            if display_step > 0 and step % display_step == 0:
                examples = gen(show_noise, critic_image_count, im_milestone)
                display_image(torch.clamp(examples, 0, 1), save_to_disk=True,
                              filename="s-{}".format(step), title="Iteration {}".format(step))

                avg_c_loss = sum(c_loss_history[-display_step:]) / display_step
                avg_g_loss = sum(g_loss_history[-display_step:]) / display_step

            pbar_description = "gen_loss: {0}  critic_loss: {1}  step: {2}".format(
                round(avg_g_loss, 5), round(avg_c_loss, 5), step)
            pbar.set_description(pbar_description, refresh=True)

            if checkpoint_step > 0 and step % checkpoint_step == 0:

                # Backup Models
                torch.save({
                    'gen': gen.state_dict(),
                    'critic': critic.state_dict(),
                    'epoch': epoch,
                    'critic_image_count': critic_image_count,
                    'im_milestone': im_milestone
                }, f'./checkpoints/chk-{step}.pth')


def test():
    gen = StyleGAN(image_progression, z_size=noise_size).to(device)
    critic = Critic(image_progression).to(device)
    fake_noise = get_truncated_noise(
        12, 512, 0.75).to(device)
    for i in range(100):
        fake = gen.forward(fake_noise, i, 10)
        end = critic.forward(fake.detach(), i, 10)


def memory_check():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

# TODO
# - Implement dynamic growth (take progression sizes)
# - implement FID checking as a progressbar stat
# - code cleanup


if __name__ == "__main__":
    if torch.device('cuda' if torch.cuda.is_available() else 'cpu').type == 'cuda':
        print(torch.cuda.get_device_name(0))
    train()
