import argparse
import os
import torch
from torch import nn
from torchvision import utils

from gan import Generator
from helper import get_truncated_noise

# Just a simple script to create a video of shifting images

if __name__ == "__main__":
    gen = nn.DataParallel(Generator().to('cuda'))
    save = torch.load('./chk-116000.pth')

    gen.load_state_dict(save["gen"])

    steps = save["step"]
    alpha = save["alpha"]

    z = get_truncated_noise(60, 512, 0.7)

    noise = []
    for i in range(8):
        size = 4 * 2 ** i
        noise.append(torch.randn(1, 1, size, size, device='cuda'))

    e = 0
    
    for i in range(59):
        start = z[i].unsqueeze(0)
        end = z[i + 1].unsqueeze(0)
        for psi in range(61):
            interpolation = torch.lerp(start, end, float(psi / 60))
            
            utils.save_image(
                gen.forward(interpolation, noise=noise, steps=steps, alpha=alpha),
                os.path.join('./output', f"image_{e + 1}.png"),
            )

            e += 1
        
    



