import torch
import torchvision
from torchvision import datasets, transforms, utils
from torch import nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
from scipy.stats import truncnorm

def display_image(images, num_display = 9, save_to_disk=False, save_dir='./output', filename="figure", title="Images"):
    if images.dim() == 3: # single image
        plt.imshow(images.permute(1, 2, 0))
    else: # multiple images, show first {num_display} in grid
        image_grid = utils.make_grid(images.detach().cpu()[:num_display], nrow=int(math.sqrt(num_display)))
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    
    plt.title(title)

    if save_to_disk:
        plt.savefig('{0}/{1}.png'.format(save_dir, filename))
    else:
        plt.show()
    

def get_truncated_noise(n_samples, z_dim, truncation):
    truncated_noise = truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim ))
    return torch.Tensor(truncated_noise)

def calculate_alphas(discriminator_count, im_milestone):

    alphas = [0 for x in range(6)] # Change range(x) to match number of alphas (# Blocks = # Alphas)

    running_count = float(discriminator_count / im_milestone) + 4
    for index, val in enumerate(alphas):
        
        if running_count < 1.0:
            continue
        elif running_count >= 1.0 and running_count <= 2.0:
            alphas[index] = round(running_count - 1, 7)
        elif running_count > 2.0:
            alphas[index] = None
            if running_count < 3.0:
                alphas[index] = 1
        
        running_count = running_count - 2
    
    if alphas[:1:-1][0] is None:
        alphas[-1] = 1
    return alphas


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
    
    def generate_mapping_block(self, in_chan: int, out_chan:int):
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
            torch.randn((1, num_channels, 1, 1))
        )
    
    def forward(self, conv_output):
        noise_shape = (conv_output.shape[0], 1, conv_output.shape[2], conv_output.shape[3])

        noise = torch.randn(noise_shape).to('cuda')

        return conv_output + (self.weights * noise)


class AdaINBlock(nn.Module):
    def __init__(self, noise_length, num_channels):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(num_channels)
        self.y_scale = nn.Linear(noise_length, num_channels)
        self.y_bias = nn.Linear(noise_length, num_channels)

        # Initialization as outlined in Whitepaper
        self.y_scale.weight.data.fill_(0) 
        self.y_bias.weight.data.fill_(0)
        self.y_scale.bias.data.fill_(1)
        self.y_bias.bias.data.fill_(0)

    def forward(self, image, noise):
        return (self.instance_norm(image) * self.y_scale(noise)[:, :, None, None]) + self.y_bias(noise)[:, :, None, None] # TODO fix?

class StyleGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_color_channels=3, noise_dim=512, image_size=(8,8), previous_image_size=(4,4), kernel_size=3):
        super().__init__()

        self.simple_upsample = nn.Upsample(image_size, mode='bilinear')

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.inject_noise_1 = InjectSecondaryNoise(out_channels)
        self.AdaIN_1 = AdaINBlock(noise_dim, out_channels)
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.inject_noise_2 = InjectSecondaryNoise(out_channels)
        self.AdaIN_2 = AdaINBlock(noise_dim, out_channels)
        self.activation_2 = nn.LeakyReLU(negative_slope=0.2)

        # These layers convert to 3 channels if we're fading in, or if this is the final layer. 

        self.large_block_to_image = nn.Conv2d(out_channels, image_color_channels, kernel_size=1) # convert data after convolutions to three color channels
        self.small_block_to_image = nn.Conv2d(in_channels, image_color_channels, kernel_size=1) # convert small data to three color channels

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

        if alpha == None: # assuming that this is not the last block
            return out
        elif alpha >= 0:
            small_image = self.small_block_to_image(x)
            small_image_upsampled = self.simple_upsample(small_image)

            conv_image = self.large_block_to_image(out)

            return torch.lerp(small_image_upsampled, conv_image, alpha)
        else:
            raise ValueError('alpha not valid!')

class StyleGAN(nn.Module):
    def __init__(self, z_size=512, image_channels=3):
        super().__init__()
        
        # Save parameters.
        self.z_size = z_size

        self.image_channels = image_channels

        # Z -> W
        self.noise_mapping = MappingLayers(z_size)

        # Synthesis network: starting constant, init to 1
        self.starting_constant = nn.Parameter(
            torch.ones((1, z_size, 4, 4))
        )


        # Constant -> 4x4
        self.block_0 = StyleGANBlock(512, 512, image_size=(4,4), image_color_channels=image_channels)

        # 4x4 -> 8x8
        self.block_1 = StyleGANBlock(512, 512, image_color_channels=image_channels)

        # 8x8 -> 16x16
        self.block_2 = StyleGANBlock(512, 512, image_size=(16,16), previous_image_size=(8,8), image_color_channels=image_channels)

        # 16x16 -> 32x32
        self.block_3 = StyleGANBlock(512, 512, image_size=(32,32), previous_image_size=(16,16), image_color_channels=image_channels)

        # 32x32 -> 64x64
        self.block_4 = StyleGANBlock(512, 256, image_size=(64,64), previous_image_size=(32,32), image_color_channels=image_channels)

        # 64x64 -> 128x128
        self.block_5 = StyleGANBlock(256, 128, image_size=(128,128), previous_image_size=(64,64), image_color_channels=image_channels)

        # Number of alphas: 6
    
    def forward(self, z_noise, discriminator_count, im_milestone):
        # calculate alphas
        alphas = calculate_alphas(discriminator_count, im_milestone)

        # forward pass https://pytorch.org/docs/master/generated/torch.nn.ModuleList.html
        w_noise = self.noise_mapping(z_noise)

        output = self.block_0(self.starting_constant, z_noise, do_upsample=False, alpha=alphas[0])

        if alphas[0] is not None:
            return output
        
        output = self.block_1(output, z_noise, alpha=alphas[1])
        if alphas[1] is not None:
            return output
        
        output = self.block_2(output, z_noise, alpha=alphas[2])
        if alphas[2] is not None:
            return output
        
        output = self.block_3(output, z_noise, alpha=alphas[3])
        if alphas[3] is not None:
            return output
        
        output = self.block_4(output, z_noise, alpha=alphas[4])
        if alphas[4] is not None:
            return output

        output = self.block_5(output, z_noise, alpha=alphas[5])

        return output


def get_generator_loss(crit_fake_pred):
    return -crit_fake_pred.mean()

class Critic(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()

        # 128 -> 64
        self.block_1 = CriticBlock(in_size=128, out_size=256)

        # 64 -> 32
        self.block_2 = CriticBlock(in_size=256)

        # 32 -> 16
        self.block_3 = CriticBlock()

        # 16 -> 8
        self.block_4 = CriticBlock()

        # 8 -> 4
        self.block_5 = CriticBlock()

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
    
    def forward(self, x, discriminator_count, im_milestone):
        alphas = calculate_alphas(discriminator_count, im_milestone)[::-1]

        out = x

        if out.shape[2] == 128 and out.shape[3] == 128:
            out = self.block_1(out, alpha=alphas[0])
        
        if out.shape[2] == 64 and out.shape[3] == 64:
            out = self.block_2(out, alpha=alphas[1])

        if out.shape[2] == 32 and out.shape[3] == 32:
            out = self.block_3(out, alpha=alphas[2])
        
        if out.shape[2] == 16 and out.shape[2] == 16:
            out = self.block_4(out, alpha=alphas[3])
        
        if out.shape[2] == 8 and out.shape[2] == 8:
            out = self.block_5(out, alpha=alphas[4])
        elif alphas[4] == 0:
            out = self.from_rgb(out)


        return self.final_layers(out)
    
    def get_critic_loss(self, crit_fake_pred, crit_real_pred, epsilon, real_images, fake_images, discriminator_count, im_milestone, c_lambda):
        mixed_images = real_images * epsilon + (1 - epsilon) * fake_images
        mixed_image_scores = self.forward(mixed_images, discriminator_count, im_milestone)

        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_image_scores,
            grad_outputs=torch.ones_like(mixed_image_scores), 
            create_graph=True,
            retain_graph=True,
        )[0]

        # create gradient penalty

        gradient = gradient.view(len(gradient), -1)

        gradient_norm = gradient.norm(2, dim=1)
        
        penalty = ((gradient_norm - 1)**2).mean()

        # put it all together

        diff = -(crit_real_pred.mean() - crit_fake_pred.mean())

        gp = c_lambda * penalty

        return diff + gp

class MiniBatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('Tensor dim must be 4, got {} instead.'.format(x.dim()))

        return torch.cat((x, torch.std(x[:, None], dim=2)), 1)

class CriticBlock(nn.Module):
    def __init__(self, image_channels=3, in_size=512, out_size=512, leaky_relu_negative_slope=0.2):
        super().__init__()

        self.from_rgb = nn.Conv2d(image_channels, in_size, 1)

        self.main_block = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.LeakyReLU(leaky_relu_negative_slope),
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.LeakyReLU(leaky_relu_negative_slope),
        )

        self.downsample = nn.AvgPool2d(2)
    
    def forward(self, x, alpha=None):
        if alpha is None:
            return self.downsample(self.main_block(x))
        elif alpha >= 0:

            conv_out = self.downsample(self.main_block(self.from_rgb(x)))

            simple_downsample = self.from_rgb(self.downsample(x))

            return torch.lerp(simple_downsample, conv_out, alpha)


class Debug(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

# IMPORTANT CONSTANTS
batch_size = 24 # Image batch size; Depends on VRAM available
im_milestone=(50 * 1000) # Progressive Growth block fade in constant; Each progression (fade-in/stabilization period) lasts X images 
c_lambda = 10 # WGAN-GP Gradient Penalty coefficient
noise_size = 512
device = 'cuda'
beta_1 = 0 # For Adam optimizer
beta_2 = 0.99 # For Adam optimizer
learning_rate = 0.0001

num_epochs = 50
critic_repeats = 1 # per Kerras et al

display_step = 50

gen_weight_decay = 0.999

# Initialize Generator
gen = StyleGAN(z_size=noise_size).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2), weight_decay=gen_weight_decay)

# Initialize Critic
critic = Critic().to(device)
critic_opt = torch.optim.Adam(critic.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

# LOADING DATA
intel_image_transformation = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.ConvertImageDtype(float),
    transforms.Resize((128, 128))
])

glacier_images = datasets.ImageFolder('./data/glaciers', intel_image_transformation)
building_images = datasets.ImageFolder('./data/buildings', intel_image_transformation)
forest_images = datasets.ImageFolder('./data/forest', intel_image_transformation)
anime_images = datasets.ImageFolder('./data/anime', intel_image_transformation)

images = torch.utils.data.DataLoader(anime_images, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

# Create a constant set of noise vectors to show same image progression
show_noise = get_truncated_noise(9, 512, 0.75).to(device)

# Training Loop

critic_image_count = 0
step = 0
for epoch in range(num_epochs):

    pbar = tqdm(images)

    for x, _ in pbar:
        current_batch_size = len(x)

        for i in range(critic_repeats):
            critic_opt.zero_grad()
            fake_noise = get_truncated_noise(current_batch_size, noise_size, 0.75).to(device)

            fake_images = gen.forward(fake_noise, critic_image_count, im_milestone)

            x = torch.nn.functional.interpolate(x, size=(fake_images.shape[2], fake_images.shape[3]), mode='bilinear').to(device, dtype=torch.float)

            critic_fake_pred = critic.forward(fake_images.detach(), critic_image_count, im_milestone)

            critic_real_pred = critic.forward(x, critic_image_count, im_milestone)

            epsilon = torch.rand(len(x), 1, 1, 1, device=device, requires_grad=True)

            loss = critic.get_critic_loss(critic_fake_pred, critic_real_pred, epsilon, x, fake_images, critic_image_count, im_milestone, c_lambda)

            loss.backward(retain_graph=True)

            critic_opt.step()

            critic_image_count += current_batch_size
            
        
        gen_opt.zero_grad()
        more_fake_noise = get_truncated_noise(current_batch_size, noise_size, 0.75).to(device)
        more_fake_images = gen(more_fake_noise, critic_image_count, im_milestone)

        critic_fake_pred = critic(more_fake_images, critic_image_count, im_milestone)

        gen_loss = get_generator_loss(critic_fake_pred)
        gen_loss.backward()

        gen_opt.step()

        step += 1

        if display_step > 0 and step % display_step == 0:
            examples = gen(show_noise, critic_image_count, im_milestone)
            display_image(torch.clamp(examples, 0, 1), save_to_disk=True, filename="s-{}".format(step), title="Iteration {}".format(step))


        pbar_description = "gen_loss: {0}  critic_loss: {1}  step: {2}".format(round(gen_loss.item(),5), round(loss.item(),5), step)
        pbar.set_description(pbar_description, refresh=True)





# TODO
# - Implement dynamic growth (take progression sizes)
# - implement FID checking as a progressbar stat
# - checkpointing every x iterations
# - Pbar format
# - code cleanup

# Make Critic block image in and out size a property


