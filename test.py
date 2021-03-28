import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn

# a = torch.ones((3,2), requires_grad=True)

# b = 4*((a - 3)**2) + 6.2

# print(a)
# print(b)

# print(torch.lerp(a , b, .9))

a = torch.randn((3,3))

print(a.size())

print(a[:, None].size())