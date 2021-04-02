import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn

def calculate_alphas(discriminator_count, num_images_fade_in):

    alphas = [0 for x in range(6)] # Change range(x) to match number of alphas (# Blocks = # Alphas)

    running_count = float(discriminator_count / num_images_fade_in) + 2
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

# size = 1024
# 

# d = torch.randn((4, 512, 4, 4))
# # print(torch.std(d[:, None], dim=2).shape)

# e = torch.cat((d, torch.std(d[:, None], dim=2)), 1)
# print(e.shape)

for i in range(1000):
    alphas = calculate_alphas(i, 100)
    if i % 25 == 0:
        print(alphas)