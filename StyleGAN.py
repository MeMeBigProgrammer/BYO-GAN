import torch
import torchvision
from torchvision import datasets, transforms, utils
from torch import nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math

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



class MappingLayers(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=1024): # The dimensions should remain 1:1 for z -> w. 
        super.__init__()
        self.layers = nn.Sequential(
            generate_mapping_block(in_channels, hidden_channels),
            generate_mapping_block(hidden_channels, hidden_channels),
            generate_mapping_block(hidden_channels, hidden_channels),
            generate_mapping_block(hidden_channels, hidden_channels),
            generate_mapping_block(hidden_channels, hidden_channels),
            generate_mapping_block(hidden_channels, hidden_channels),
            generate_mapping_block(hidden_channels, hidden_channels),
            nn.Linear(hidden_channels, in_channels)
        )
    
    def generate_mapping_block(in_chan: int, out_chan:int):
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
    def __init__(self):
        super().__init__()

        self.y_scale = nn.Linear()
        self.y_bias = nn.Linear()

        # Instance norm too

for x, _ in tqdm(images):
    display_image(x)
