import argparse
import os
import torch
from torch import nn
from torchvision import utils

from gan import Generator
from helper import get_truncated_noise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to saved model", type=str)
    parser.add_argument("images", help="number of images to produce", type=int)
    parser.add_argument(
        "-d" "--device",
        help="specify pytorch device to run model on",
        dest="device",
        default="cuda",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        help="output directory",
        default=".",
        type=str,
    )
    parser.add_argument(
        "-z" "--z-size",
        dest="z_size",
        help="noise size",
        default=512,
        type=int,
    )
    parser.add_argument(
        "-t" "--truncation",
        dest="trunc",
        help="truncation boundary",
        default=0.75,
        type=float,
    )
    args = parser.parse_args()

    if args.output_dir is not None and not os.path.exists(args.output_dir):
        raise OSError("path does not exist!")

    gen = nn.DataParallel(Generator().to(args.device))

    save = torch.load(args.model)

    gen.load_state_dict(save["gen"])

    for i in range(args.images):
        noise = get_truncated_noise(1, args.z_size, args.trunc)
        utils.save_image(
            gen.forward(noise, steps=save["step"], alpha=save["alpha"]),
            os.path.join(args.output_dir, f"image_{i + 1}.png"),
        )
