import configparser
import torch
import argparse

from train import train

if __name__ == "__main__":
    # Check that Cuda driver and devices are good.
    if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
        print(torch.cuda.get_device_name(0))

    # Load config section from CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Specify which config to use", type=str)
    parser.add_argument(
        "-c", "--checkpoint", dest="checkpoint", help="Relative path to checkpoint file"
    )
    args = parser.parse_args()

    # Load config.
    config = configparser.ConfigParser()
    config.read("config.txt")

    settings = config[args.config]

    train(settings, checkpoint=args.checkpoint)
