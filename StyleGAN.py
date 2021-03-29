import torch
import torchvision
from torchvision import datasets, transforms, utils
from torch import nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
from scipy.stats import truncnorm

# IMPORTANT CONSTANTS
batch_size = 24

# LOADING DATA
intel_image_transformation = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.ConvertImageDtype(float),
    transforms.Resize((128, 128))
])

glacier_images = datasets.ImageFolder('./data/glaciers', intel_image_transformation)
building_images = datasets.ImageFolder('./data/buildings', intel_image_transformation)
forest_images = datasets.ImageFolder('./data/forest', intel_image_transformation)

images = torch.utils.data.DataLoader(glacier_images, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)


def display_image(images, num_display = 9, save_to_disk=False, save_dir='./output', filename="figure"):
    if images.dim() == 3: # single image
        plt.imshow(images.permute(1, 2, 0))
    else: # multiple images, show first {num_display} in grid
        image_grid = utils.make_grid(images.detach().cpu()[:num_display], nrow=int(math.sqrt(num_display)))
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    
    if save_to_disk:
        plt.savefig('{0}/{1}.png'.format(save_dir, filename))
        return
    
    plt.show()

def get_truncated_noise(n_samples, z_dim, truncation):
    truncated_noise = truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim ))
    return torch.Tensor(truncated_noise)


class MappingLayers(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=1024): # The dimensions should remain 1:1 for z -> w. 
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

        noise = torch.randn(noise_shape)

        return conv_output + (self.weights * noise)


class AdaINBlock(nn.Module):
    def __init__(self, noise_length, num_channels):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(num_channels)
        self.y_scale = nn.Linear(noise_length, num_channels)
        self.y_bias = nn.Linear(noise_length, num_channels)

    def forward(self, image, noise):
        return (self.instance_norm(image) * self.y_scale(noise)[:, :, None, None]) + self.y_bias(noise)[:, :, None, None] # TODO fix?

class StyleGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_color_channels=3, noise_dim=512, image_size=(8,8), previous_image_size=(4,4), kernel_size=3):
        super().__init__()

        self.simple_upsample = nn.Upsample(image_size, mode='bilinear') # TODO, wrap in one nn.sequential?

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
        
    def upsample_image(self, image, output_size): #input must have three color channels
        return F.interpolate(image, size=output_size, mode='bilinear')
    
    def mix_images(self, upsampled, convolution_output, alpha):
        return None

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
    def __init__(self, z_size=512, image_channels=3, num_images_fade_in=(800 * 1000)):
        super().__init__()
        
        # Save parameters.
        self.z_size = z_size

        self.image_channels = image_channels

        self.num_images_fade_in = num_images_fade_in

        # Z -> W
        self.noise_mapping = MappingLayers(z_size)

        # Synthesis network: starting constant
        self.starting_constant = nn.Parameter(
            torch.randn((1, z_size, 4, 4))
        )

        # Constant -> 4x4
        self.block_0 = StyleGANBlock(512, 512, image_size=(4,4))

        # 4x4 -> 8x8
        self.block_1 = StyleGANBlock(512, 512)

        # 8x8 -> 16x16
        self.block_2 = StyleGANBlock(512, 512, image_size=(16,16), previous_image_size=(8,8))

        # 16x16 -> 32x32
        self.block_3 = StyleGANBlock(512, 512, image_size=(32,32), previous_image_size=(16,16))

        # 32x32 -> 64x64
        self.block_4 = StyleGANBlock(512, 256, image_size=(64,64), previous_image_size=(32,32))

        # 64x64 -> 128x128
        self.block_5 = StyleGANBlock(256, 128, image_size=(128,128), previous_image_size=(64,64))

        # Number of alphas: 6
    
    def forward(self, z_noise, discriminator_count):
        # calculate alphas
        alphas = self.calculate_alphas(discriminator_count)

        # forward pass
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

    def calculate_alphas(self, discriminator_count):

        alphas = [0 for x in range(6)] # Change this number to match number of alphas

        running_count = float(discriminator_count / self.num_images_fade_in) + 2
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


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

for x, _ in tqdm(images):
    print()
