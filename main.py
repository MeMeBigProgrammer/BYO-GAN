import configparser
import torch

from train import train

# LOADING DATA
config_section = "anime"

if __name__ == "__main__":
    # Check Cuda Driver and Devices are good.
    if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
        print(torch.cuda.get_device_name(0))

    # Load config file.
    config = configparser.ConfigParser()
    config.read("config.txt")

    settings = config[config_section]

    train(settings, checkpoint=None)
