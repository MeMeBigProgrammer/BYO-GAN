import torch
from torchvision import datasets, transforms

from train import train

"""
TODOs:
- CLI arguments
- gan.py cleanup
    Remove PixelNorm?
- Better Logging
"""

final_image_size = 512

# LOADING DATA
transformation = transforms.Compose(
    [
        transforms.Resize((final_image_size, final_image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        transforms.ConvertImageDtype(float),
    ]
)

anime_images = datasets.ImageFolder("./data/anime", transformation)
art_images = datasets.ImageFolder("./data/art", transformation)


# 4, 8, 16, 32, 64, 128, 256, 512
epoch_progresson = [1, 40, 32, 30, 25, 25, 20, 15]

batch_progression = [24, 24, 24, 20, 16, 10, 5, 5]

fade_percentage = 0.5  # Percentage of each step will be a fade in.

if __name__ == "__main__":
    if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
        print(torch.cuda.get_device_name(0))

    train(anime_images, epoch_progresson, batch_progression, fade_percentage)
