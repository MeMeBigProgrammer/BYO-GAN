import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
import math

# https://github.com/SiskonEmilia/StyleGAN-PyTorch/blob/master/train.py
# https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb

a = torch.arange(24).view(6, 4)

b = a.view((-1, 2, 2))

print(a, '\n\n', b)
