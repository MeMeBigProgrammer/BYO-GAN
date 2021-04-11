import sys
import torch
import torchvision
import gc
import math
from torch import nn
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.stats import truncnorm
from datetime import datetime


def train():
    print("Hello There!")
    gen = Generator()
    fake_noise = get_truncated_noise(4, 512, 0.75)
    a = gen.forward(fake_noise)
    critic = Critic()
    print(critic.forward(a))