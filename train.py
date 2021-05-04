import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import datasets, transforms
import os

from gan import Generator, Critic
from helper import (
    get_truncated_noise,
    display_image,
    set_requires_grad,
)


def train(config, checkpoint=None):

    # Load constants.
    c_lambda = int(config.get("gradient_lambda", 10))
    noise_size = int(config.get("noise_length", 512))
    device = config.get("device", "cuda")
    beta_1 = float(config.get("beta_1", 0.00))
    beta_2 = float(config.get("beta_2", 0.99))
    learning_rate = float(config.get("lr", 0.001))
    critic_repeats = int(config.get("critic_repeats", 1))
    use_r1_loss = str(config.get("use_r1", "True")) == "True"
    num_workers = int(config.get("dataloader_threads", 2))

    display_step = int(config.get("display_step", 250))
    checkpoint_step = int(config.get("checkpoint_step", 2000))
    refresh_stat_step = int(config.get("refresh_stat_step", 5))

    # The batch size in each image size progression.
    batch_progression = config.get("batch_progression").split(",")
    batch_progression = list(map(int, batch_progression))

    # The number of epochs in each image size progression.
    epoch_progresson = config.get("epoch_progression").split(",")
    epoch_progresson = list(map(int, epoch_progresson))

    # Percentage of each step that will be a fade in.
    fade_in_percentage = float(config.get("fade_percentage", 0.5))

    transformation = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            transforms.ConvertImageDtype(float),
        ]
    )

    # Path to dataset.
    data_path = config.get("data", None)
    if not os.path.exists(os.path.join(data_path, "prepared")):
        raise OSError("Did not detect prepared dataset!")

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
    gen = nn.DataParallel(gen)
    gen.train()

    # Initialize Critic
    critic = Critic().to(device)
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
    )
    critic = nn.DataParallel(critic)
    critic.train()

    # Create a constant set of noise vectors to show image progression.
    show_noise = get_truncated_noise(25, 512, 0.75).to(device)

    # Some other variables to track.
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
        last_epoch = save["epoch"]
    else:
        last_step = None
        last_epoch = None

    for index, step_epochs in enumerate(epoch_progresson):

        if last_step is not None and index + 1 < last_step:
            continue

        steps = int(index + 1)
        im_count = 0
        images = datasets.ImageFolder(
            os.path.join(data_path, "prepared", f"set_{steps}"), transformation
        )
        dataset = torch.utils.data.DataLoader(
            images,
            batch_size=batch_progression[index],
            shuffle=True,
            num_workers=num_workers,
        )

        fade_in = fade_in_percentage * step_epochs * len(dataset)

        print(f"STARTING STEP #{steps}")

        for epoch in range(step_epochs):

            if last_epoch is not None and epoch < last_epoch:
                continue
            else:
                last_epoch = None

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

                        c_loss = critic.module.get_r1_loss(
                            critic_fake_pred,
                            critic_real_pred,
                            real_im,
                            fake_im,
                            steps,
                            alpha,
                            c_lambda,
                        )
                    else:
                        c_loss = critic.module.get_wgan_loss(
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

                    g_loss = gen.module.get_r1_loss(critic_fake_pred)

                else:

                    g_loss = gen.module.get_wgan_loss(critic_fake_pred)

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
                        f"g_loss: {avg_g_loss:.3}  c_loss: {avg_c_loss:.3}  epoch: {epoch + 1}",
                        refresh=True,
                    )

                with torch.no_grad():
                    examples = gen(show_noise, alpha=alpha, steps=steps)
                    if iters > 0 and iters % display_step == 0:
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
                                "epoch": epoch,
                                "alpha": alpha,
                            },
                            f"./checkpoints/chk-{iters}.pth",
                        )

    # TRAINING FINISHED - save final set of samples and save model.
    examples = gen(show_noise, alpha=alpha, steps=steps)
    torch.save(
        {
            "gen": gen.state_dict(),
            "critic": critic.state_dict(),
            "iter": iters,
            "im_count": im_count,
            "step": steps,
            "epoch": epoch,
            "alpha": alpha,
        },
        "./checkpoints/FINAL.pth",
    )
    print("TRAINING IS FINISHED - MODEL SAVED!")
