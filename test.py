import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn

# def calculate_alphas(num_images_fade_in, discriminator_count):

#     alphas = [0 for x in range(6)] # Change this number to match number of alphas

#     running_count = float(discriminator_count / num_images_fade_in) + 2
#     for index, val in enumerate(alphas):
        
#         if running_count < 1.0:
#             continue
#         elif running_count >= 1.0 and running_count <= 2.0:
#             alphas[index] = round(running_count - 1, 7)
#         elif running_count > 2.0:
#             alphas[index] = None
#             if running_count < 3.0:
#                 alphas[index] = 1
        
#         running_count = running_count - 2
    
#     if alphas[:1:-1][0] is None:
#         alphas[-1] = 1
#     return alphas


# size = 1024
# 

# d = torch.randn((4, 512, 4, 4))
# # print(torch.std(d[:, None], dim=2).shape)

# e = torch.cat((d, torch.std(d[:, None], dim=2)), 1)
# print(e.shape)

a = torch.ones((4,4))
b = torch.zeros((4,4))

print(torch.lerp(b, a, .6))