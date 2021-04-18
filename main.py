import configparser

import torch
from torchvision import datasets, transforms

from train import train

"""
TODOs:
https://docs.python.org/3/library/configparser.html
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

# Check config file for different presets.
config_section = "anime"

if __name__ == "__main__":
    # Check Cuda Driver and Devices are good.
    if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
        print(torch.cuda.get_device_name(0))

    # Load config file.
    config = configparser.ConfigParser()
    config.read("config.txt")

    settings = config[config_section]

    # The batch size in each image size progression.
    batch_progression = settings.get(
        "batch_progression", "24,16,16,16,12,10,5,5"
    ).split(",")
    batch_progression = list(map(int, batch_progression))

    # The number of epochs in each image size progression.
    epoch_progresson = settings.get(
        "epoch_progression", "10,20,20,30,30,20,20,15"
    ).split(",")
    epoch_progresson = list(map(int, epoch_progresson))

    # Percentage of each step will be a fade in.
    fade_percentage = float(settings.get("fade_percentage", 0.5))

    # Path to dataset.
    data_path = settings.get("data", None)

    if data_path is None:
        raise ValueError("Data path cannot be NoneType!")

    images = datasets.ImageFolder(data_path, transformation)

    train(
        images,
        epoch_progresson,
        batch_progression,
        fade_percentage,
        checkpoint=None,
    )
