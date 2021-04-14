import torch
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from gan import Generator, Critic, EMA
from helper import (
    get_truncated_noise,
    display_image,
    get_progression_step,
    set_requires_grad,
    save_image_samples,
)

"""
TODOs:
- CLI arguments
- gan.py cleanup
    Remove PixelNorm?
- reimplement WGAN-GP
- Better Logging
"""

# IMPORTANT CONSTANTS
batch_size = 24
# Progressive Growth block fade in constant; Each progression (fade-in/stabilization period) lasts X images
im_milestone = 125 * 1000
c_lambda = 10
noise_size = 512
device = "cuda"
beta_1 = 0
beta_2 = 0.99
learning_rate = 0.002
critic_repeats = 1

num_epochs = 500
display_step = 250
checkpoint_step = 2000
refresh_stat_step = 5

final_image_size = 512

# Create a constant set of noise vectors to show same image progression.
show_noise = get_truncated_noise(25, 512, 0.75).to(device)

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

images = torch.utils.data.DataLoader(
    anime_images, batch_size=batch_size, shuffle=True, num_workers=2
)


def train(checkpoint=None):
    # Initialize Generator
    gen = Generator().to(device)
    gen_opt = torch.optim.Adam(
        [
            {
                "params": gen.to_w_noise.parameters(),
                "lr": (learning_rate * 0.01),
            },
            {"params": gen.gen_blocks.parameters()},
            {"params": gen.to_rgbs.parameters()},
        ],
        lr=learning_rate,
        betas=(beta_1, beta_2),
    )
    ema = EMA(gen, 0.99)
    ema.register()
    gen.train()

    # Initialize Critic
    critic = Critic().to(device)
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
    )
    critic.train()

    im_count = 2 * im_milestone
    iters = 0
    c_loss_history = []
    g_loss_history = []

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

                real_im = (
                    torch.nn.functional.interpolate(
                        real_im,
                        size=(fake_im.shape[2], fake_im.shape[3]),
                        mode="bilinear",
                    )
                    .to(device, dtype=torch.float)
                    .requires_grad_()
                )

                critic_fake_pred = critic(fake_im.detach(), steps, alpha)

                critic_real_pred = critic(real_im, steps, alpha)

                critic.zero_grad()

                c_loss = critic.get_r1_loss(
                    critic_fake_pred,
                    critic_real_pred,
                    real_im,
                    fake_im,
                    steps,
                    alpha,
                    c_lambda,
                )

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
            ema.update()

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
                    f"g_loss: {avg_g_loss:.3}  c_loss: {avg_c_loss:.3}  im_c: {im_count}",
                    refresh=True,
                )

            ema.apply_shadow()
            if iters > 0 and iters % display_step == 0:
                ema.apply_shadow()
                with torch.no_grad():
                    examples = gen(show_noise, alpha=alpha, steps=steps)
                    display_image(
                        torch.clamp(examples, 0, 1),
                        save_to_disk=True,
                        filename="s-{}".format(iters),
                        title="Iteration {}".format(iters),
                        num_display=25,
                    )

            if iters > 0 and iters % checkpoint_step == 0:
                torch.save(
                    {
                        "gen": gen.state_dict(),
                        "critic": critic.state_dict(),
                        "iter": iters,
                        "im_count": im_count,
                        "exponential_average_shadow": ema.state_dict(),
                    },
                    f"./checkpoints/chk-{iters}.pth",
                )

            ema.restore()


if __name__ == "__main__":
    if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
        print(torch.cuda.get_device_name(0))

    train(checkpoint=None)
