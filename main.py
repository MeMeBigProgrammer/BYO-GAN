import sys, gc, math
from datetime import datetime
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from gan import Generator, Critic
from utils import get_truncated_noise, display_image, get_progression_step


# IMPORTANT CONSTANTS
batch_size = 24
# Progressive Growth block fade in constant; Each progression (fade-in/stabilization period) lasts X images
im_milestone = 200 * 1000
c_lambda = 10
noise_size = 512
device = "cuda"
beta_1 = 0
beta_2 = 0.99
learning_rate = 0.0001
critic_repeats = 1
gen_weight_decay = 0.999

num_epochs = 500
display_step = 50
checkpoint_step = 1000

final_image_size = 512

# Create a constant set of noise vectors to show same image progression.
show_noise = get_truncated_noise(4, 512, 0.75).to(device)

# LOADING DATA
transformation = transforms.Compose(
    [
        transforms.Resize((final_image_size, final_image_size)),
        transforms.CenterCrop((final_image_size, final_image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ConvertImageDtype(float),
    ]
)

path_root = "./data"

anime_images = datasets.ImageFolder(path_root + "/anime", transformation)

images = torch.utils.data.DataLoader(
    anime_images, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=3
)


def train():
    # Initialize Generator
    gen = Generator()
    gen_opt = torch.optim.Adam(
        [
            {"params": gen.to_w_noise.parameters(), "lr": (learning_rate * 0.01)},
            {"params": gen.gen_blocks.parameters()},
            {"params": gen.to_rgbs.parameters()},
        ],
        lr=learning_rate,
        betas=(beta_1, beta_2),
        weight_decay=gen_weight_decay,
    )

    # Initialize Critic
    critic = Critic().to(device)
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
    )

    im_count = 0
    c_loss_history = []
    g_loss_history = []
    iters = 0

    for epoch in range(0):

        pbar = tqdm(images)

        for real_im, _ in pbar:
            cur_batch_size = len(real_im)

            for i in range(critic_repeats):
                critic_opt.zero_grad()
                z_noise = get_truncated_noise(cur_batch_size, noise_size, 0.75).to(
                    device
                )

                alpha, steps = get_progression_step(im_count, im_milestone)

                fake_im = gen(z_noise, steps=steps, alpha=alpha)

                real_im = torch.nn.functional.interpolate(
                    real_im,
                    size=(fake_im.shape[2], fake_im.shape[3]),
                    mode="bilinear",
                ).to(device, dtype=torch.float)

                critic_fake_pred = critic(fake_im, steps=steps, alpha=alpha)

                critic_real_pred = critic(real_im, steps=steps, alpha=alpha)

                epsilon = torch.rand(
                    cur_batch_size, 1, 1, 1, device=device, requires_grad=True
                )

                loss = critic.get_loss(
                    critic_fake_pred,
                    critic_real_pred,
                    epsilon,
                    real_im,
                    fake_im,
                    steps,
                    alpha,
                    c_lambda,
                )

                loss.backward(retain_graph=True)

                critic_opt.step()

                im_count += cur_batch_size

                c_loss_history.append(loss.item())

            gen_opt.zero_grad()
            noise = get_truncated_noise(cur_batch_size, noise_size, 0.75).to(device)
            fake_images = gen(noise, im_count, im_milestone)

            alpha, steps = get_progression_step(im_count, im_milestone)

            critic_fake_pred = critic(fake_images, steps=steps, alpha=alpha)

            gen_loss = gen.get_loss(critic_fake_pred)
            gen_loss.backward()

            gen_opt.step()

            g_loss_history.append(gen_loss.item())

            iters += 1


if __name__ == "__main__":
    if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
        print(torch.cuda.get_device_name(0))

    train()
