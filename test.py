import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
import math

# https://github.com/SiskonEmilia/StyleGAN-PyTorch/blob/master/train.py
# https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb

a = torch.zeros((4, 3, 2, 2))

b = a + 1e-8

print(a, b)
