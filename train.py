import torch
from tqdm.auto import tqdm

from gan import Generator, Critic, EMA
from helper import (
    get_truncated_noise,
    display_image,
    set_requires_grad,
    save_image_samples,
)

# IMPORTANT CONSTANTS
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


def train(
    images,
    epoch_progresson,
    batch_progression,
    fade_in_percentage,
    checkpoint=None,
    use_r1_loss=True,
):

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

    # Create a constant set of noise vectors to show same image progression.
    show_noise = get_truncated_noise(25, 512, 0.75).to(device)

    # Some other variables to keep track of.
    iters = 0
    c_loss_history = []
    g_loss_history = []

    if checkpoint is not None:
        save = torch.load(checkpoint)
        gen.load_state_dict(save["gen"])
        critic.load_state_dict(save["critic"])
        iters = save["iter"]
        im_count = save["im_count"]
        last_step = save["step"]
    else:
        last_step = None

    for index, step_epochs in enumerate(epoch_progresson):

        if last_step is not None and index + 1 < last_step:
            continue

        steps = int(index + 1)
        im_count = 0
        dataset = torch.utils.data.DataLoader(
            images,
            batch_size=batch_progression[index],
            shuffle=True,
            num_workers=2,
        )

        fade_in = fade_in_percentage * step_epochs * len(dataset)

        print(f"STARTING STEP #{steps}")

        for epoch in range(step_epochs):

            pbar = tqdm(dataset)

            for real_im, _ in pbar:
                cur_batch_size = len(real_im)

                set_requires_grad(critic, True)
                set_requires_grad(gen, False)

                for i in range(critic_repeats):
                    z_noise = get_truncated_noise(cur_batch_size, noise_size, 0.75).to(
                        device
                    )

                    alpha = im_count / fade_in

                    if alpha > 1.0:
                        alpha = None

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

                    if use_r1_loss:

                        c_loss = critic.get_r1_loss(
                            critic_fake_pred,
                            critic_real_pred,
                            real_im,
                            fake_im,
                            steps,
                            alpha,
                            c_lambda,
                        )
                    else:
                        c_loss = critic.get_wgan_loss(
                            critic_fake_pred,
                            critic_real_pred,
                            real_im,
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

                alpha = im_count / fade_in

                if alpha > 1.0:
                    alpha = None

                fake_images = gen(noise, steps=steps, alpha=alpha)

                critic_fake_pred = critic(fake_images, steps, alpha)

                if use_r1_loss:

                    g_loss = gen.get_r1_loss(critic_fake_pred)

                else:

                    g_loss = gen.get_wgan_loss(critic_fake_pred)

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
                            "step": steps,
                        },
                        f"./checkpoints/chk-{iters}.pth",
                    )

                ema.restore()
