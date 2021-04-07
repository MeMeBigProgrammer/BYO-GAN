import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
import math

# https://github.com/SiskonEmilia/StyleGAN-PyTorch/blob/master/train.py
# https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb

m = nn.Parameter(torch.ones(1, 512, 4, 4))

print(m(4))
