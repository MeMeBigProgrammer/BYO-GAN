import argparse
import os
import torch
from torch import nn
from torchvision import utils

from gan import Generator
from helper import get_truncated_noise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to checkpoint", type=str)
    parser.add_argument("images", help="number of images to produce", type=int)
    parser.add_argument(
        "device", help="device to run the program on. (cuda/CPU)", type=str
    )
    parser.add_argument(
        "-d", "--output", dest="output_dir", help="dir for output", type=str
    )
    args = parser.parse_args()

    if args.output_dir is not None and not os.path.exists(args.output_dir):
        raise OSError("path does not exist!")
    elif args.output_dir is None:
        args.output_dir = "."

    gen = nn.DataParallel(Generator().to(args.device))

    save = torch.load(args.model)

    gen.load_state_dict(save["gen"])
    steps = save["step"]

    noise = get_truncated_noise(args.images, 512, 0.75)

    output = gen.forward(noise, steps=save["step"], alpha=0.01)

    for index, image in enumerate(output):
        utils.save_image(image, os.path.join(args.output_dir, f"image_{index + 1}.png"))
