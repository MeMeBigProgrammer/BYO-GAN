import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn

def calculate_alphas(num_images_fade_in, discriminator_count):

    alphas = [0 for x in range(6)]

    running_count = float(discriminator_count / num_images_fade_in) + 2
    for index, val in enumerate(alphas):
        
        if running_count < 1.0:
            return alphas
        elif running_count >= 1.0 and running_count <= 2.0:
            alphas[index] = round(running_count - 1, 7)
        elif running_count > 2.0:
            alphas[index] = None
            if running_count < 3.0:
                alphas[index] = 1
        
        running_count = running_count - 2
    
        

a = [0, 0, None]
print(a[:1:-1][0] is None)
a[-1] = 0
print(a)