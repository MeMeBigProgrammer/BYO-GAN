import sys, gc, math
from datetime import datetime
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from gan import Generator, Critic
from utils import (
    get_truncated_noise,
    display_image,
    get_progression_step,
    set_requires_grad,
)

# IMPORTANT CONSTANTS
batch_size = 32
# Progressive Growth block fade in constant; Each progression (fade-in/stabilization period) lasts X images
im_milestone = 110 * 1000
c_lambda = 10
noise_size = 512
device = "cuda"
beta_1 = 0
beta_2 = 0.99
learning_rate = 0.0002
critic_repeats = 1
gen_weight_decay = 0.999

num_epochs = 500
display_step = 100
checkpoint_step = 1000

final_image_size = 512

refresh_stat_step = 5

# Create a constant set of noise vectors to show same image progression.
show_noise = get_truncated_noise(4, 512, 0.75).to(device)

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

images = torch.utils.data.DataLoader(
    anime_images, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=3
)


def train(checkpoint=None):
    # Initialize Generator
    gen = Generator().to(device)
    gen_opt = torch.optim.Adam(
        [
            {
                "params": gen.to_w_noise.parameters(),
                "lr": (learning_rate * 0.01),
                "mult": 0.01,
            },
            {"params": gen.gen_blocks.parameters()},
            {"params": gen.to_rgbs.parameters()},
        ],
        lr=learning_rate,
        betas=(beta_1, beta_2),
        weight_decay=gen_weight_decay,
    )
    gen.train()

    # Initialize Critic
    critic = Critic().to(device)
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
    )
    critic.train()

    im_count = 1 * im_milestone
    iters = 0
    c_loss_history = []
    g_loss_history = []

    show_noise = get_truncated_noise(16, 512, 0.75).to(device)

    if checkpoint is not None:
        save = torch.load(checkpoint)
        gen.load_state_dict(save["gen"])
        critic.load_state_dict(save["critic"])
        iters = save["iter"]
        im_count = save["im_count"]

    for epoch in range(num_epochs):

        pbar = tqdm(images)

        for real_im, _ in pbar:
            cur_batch_size = len(real_im)

            set_requires_grad(critic, True)
            set_requires_grad(gen, False)

            for i in range(critic_repeats):
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

                critic_fake_pred = critic(fake_im.detach(), steps=steps, alpha=alpha)

                c_loss = critic.get_r1_loss(critic_fake_pred, real_im.detach(), steps, alpha)

                critic.zero_grad()

                c_loss.backward(retain_graph=True)

                critic_opt.step()

                im_count += cur_batch_size

                c_loss_history.append(c_loss.item())

            set_requires_grad(critic, False)
            set_requires_grad(gen, True)

            noise = get_truncated_noise(cur_batch_size, noise_size, 0.75).to(device)

            alpha, steps = get_progression_step(im_count, im_milestone)

            fake_images = gen(noise, steps=steps, alpha=alpha)

            critic_fake_pred = critic(fake_images, steps=steps, alpha=alpha)

            g_loss = gen.get_r1_loss(critic_fake_pred)

            gen.zero_grad()
            g_loss.backward()
            gen_opt.step()

            g_loss_history.append(g_loss.item())

            iters += 1

            if iters > 0 and iters % refresh_stat_step == 0:
                avg_c_loss = (
                    sum(c_loss_history[-refresh_stat_step:]) / refresh_stat_step
                )
                avg_g_loss = (
                    sum(g_loss_history[-refresh_stat_step:]) / refresh_stat_step
                )

                pbar.set_description(
                    "g_loss: {0:.3}   c_loss: {1:.3}".format(avg_g_loss, avg_c_loss),
                    refresh=True,
                )

            if iters > 0 and iters % display_step == 0:
                with torch.no_grad():
                    examples = gen(show_noise, alpha=alpha, steps=steps)
                    display_image(
                        torch.clamp(examples, 0, 1),
                        save_to_disk=True,
                        filename="s-{}".format(iters),
                        title="Iteration {}".format(iters),
                        num_display=16,
                    )

            if iters > 0 and iters % 2500 == 0:
                torch.save(
                    {
                        "gen": gen.state_dict(),
                        "critic": critic.state_dict(),
                        "iter": iters,
                        "im_count": im_count,
                    },
                    f"./checkpoints/chk-{iters}.pth",
                )


if __name__ == "__main__":
    if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
        print(torch.cuda.get_device_name(0))

    train(checkpoint=None)
