"""
Training utilities for Generative Adversarial Networks (GANs).

This module contains various helper functions and main training procedures for training GAN models.
It provides utilities for logging progress, saving model checkpoints, calculating Frechet Inception 
Distance (FID) score, and the main training loop.

Functions:
- `log_progresso`: Logs the progress of training to an external file with a timestamp.
- `save_checkpoint`: Saves the state of the generator, discriminator, their optimizers, and loss 
lists to a checkpoint file.
- `calculate_fid`: Computes the FID score using real and generated images.
- `train_model`: The main training loop. Handles the training process for the generator and 
discriminator, logs progress, saves sample images and model checkpoints.

Utilities:
- Methods to perform operations on the model like gradient calculations, backpropagation, and 
optimizer steps.

Dependencies:
- datetime: For timestamp creation.
- time: To measure time durations.
- numpy: For mathematical operations.
- torch and torchvision: For deep learning operations, model creation, and image saving.
- tqdm: To display a progress bar during training.

Notes:
- Adjust the training hyperparameters and steps as needed to cater to specific GAN architectures 
or variations.
- Ensure necessary error handling when dealing with file paths, directories, or non-existent 
checkpoints.
"""
import os
import datetime
import time

import numpy as np
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
from tqdm import tqdm


def log_progresso(log_file, message):
    """
    Logs the training progress to an external file with a timestamp.

    Parameters:
    - log_file (str): The path to the log file.
    - message (str): The message to be logged.

    Notes:
    - The function appends the message to the log file with a timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"

    with open(log_file, "a", encoding='utf-8') as file:
        file.write(log_entry)


def save_checkpoint(
        epoch, generator, discriminator, optim_g, optim_d,
        losses_g, losses_d, path="checkpoint.pth"
    ):
    """
    Saves the state of the generator, discriminator, their optimizers, and loss lists 
    to a checkpoint file.

    Parameters:
    - epoch (int): The current epoch number.
    - generator (torch.nn.Module): The generator model.
    - discriminator (torch.nn.Module): The discriminator model.
    - optim_g (torch.optim.Optimizer): The optimizer for the generator.
    - optim_d (torch.optim.Optimizer): The optimizer for the discriminator.
    - losses_g (list): List of generator losses.
    - losses_d (list): List of discriminator losses.
    - path (str, optional): The path to save the checkpoint. Default is "checkpoint.pth".
    """
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optim_g.state_dict(),
        'optimizer_d_state_dict': optim_d.state_dict(),
        'losses_g': losses_g,
        'losses_d': losses_d
    }, path)


def calculate_fid(images_real, images_fake, inception_model):
    """
    Computes the Frechet Inception Distance (FID) score using real and generated images.

    Parameters:
    - images_real (torch.Tensor): The real images tensor.
    - images_fake (torch.Tensor): The generated images tensor.
    - inception_model (torch.nn.Module): The pre-trained Inception model to compute the FID score.

    Returns:
    - float: The computed FID score.

    Notes:
    - The FID score measures the similarity between two sets of images. Lower values indicate the 
    two sets of images are more similar.
    """
    real_features = inception_model(images_real).detach().cpu().numpy()
    fake_features = inception_model(images_fake).detach().cpu().numpy()

    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    diff_mu = mu_real - mu_fake
    cov_mean = (sigma_real + sigma_fake) / 2
    fid = diff_mu.dot(diff_mu) + np.trace(sigma_real + sigma_fake - 2*np.sqrt(cov_mean))

    return fid


def train_model(
        inception_model,
        generator,
        discriminator,
        weights_path,
        sample_size,
        sample_dir,
        optim_g,
        optim_d,
        data_loader,
        device,
        z_dim,
        k,
        lambda_k,
        gamma,
        num_epochs,
        last_epoch,
        save_model_at,
        log_dir,
        losses_g,
        losses_d
    ):
    """
    The main training loop for GANs. This function handles the training process for the generator 
    and discriminator, logs progress, saves sample images, and model checkpoints.

     Parameters:
    - inception_model (torch.nn.Module): The pre-trained Inception model for FID computation.
    - generator (torch.nn.Module): The generator model.
    - discriminator (torch.nn.Module): The discriminator model.
    - weights_path (str): Path to save model weights.
    - sample_size (int): Number of samples to generate for visualization.
    - sample_dir (str): Directory to save generated samples.
    - optim_g (torch.optim.Optimizer): Optimizer for the generator.
    - optim_d (torch.optim.Optimizer): Optimizer for the discriminator.
    - data_loader (torch.utils.data.DataLoader): DataLoader for training data.
    - device (torch.device): Device to which the models and data are sent.
    - z_dim (int): Dimension of the latent space.
    - k (float): Balance coefficient for the discriminator's loss.
    - lambda_k (float): Learning rate for k.
    - gamma (float): Hyperparameter for balance in the loss function.
    - num_epochs (int): Total number of epochs for training.
    - last_epoch (int): The last epoch number. Default is 0 for new training sessions.
    - save_model_at (int): Interval (in epochs) to save model checkpoints.
    - log_dir (str): Directory to save training logs.
    - losses_g (list, optional): List of generator losses. Default is an empty list.
    - losses_d (list, optional): List of discriminator losses. Default is an empty list.

    Returns:
    - list, list: Lists of generator and discriminator losses across epochs.

    Notes:
    - This function performs the core training procedure for the GAN model.
    - Adjust the training hyperparameters and steps as needed to cater to specific GAN 
    architectures or variations.
    """

    fixed_noise = Variable(torch.randn(sample_size, z_dim, 1, 1)).to(device)
    fid_score = 'Unavailable'

    for epoch in range(last_epoch, num_epochs + 1):
        start_time = time.time()

        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for _, data in pbar:
            images, _ = data
            images = Variable(images).to(device)

            # Discriminator update
            z_points = Variable(torch.randn(images.size(0), z_dim, 1, 1)).to(device)
            fake_images = generator(z_points)

            real_loss = torch.mean(torch.abs(discriminator(images) - images))
            fake_loss = torch.mean(torch.abs(discriminator(fake_images.detach()) - fake_images))

            d_loss = real_loss - k * fake_loss

            discriminator.zero_grad()
            d_loss.backward()
            optim_d.step()

            # Generator update
            z_points = Variable(torch.randn(images.size(0), z_dim, 1, 1)).to(device)
            fake_images = generator(z_points)

            if inception_model:
                # FID test
                fid_score = calculate_fid(images, fake_images, inception_model)

            g_loss = torch.mean(torch.abs(discriminator(fake_images) - fake_images))

            generator.zero_grad()
            g_loss.backward()
            optim_g.step()

            # Update k
            balance = (gamma * real_loss - g_loss).item()
            k += lambda_k * balance
            k = min(max(k, 0), 1)

            pbar.set_description(
                f'Epoch {epoch}/{num_epochs}, g_loss: {g_loss.data}, d_loss: {d_loss.data}'
            )

        end_time = time.time()
        epoch_duration = end_time - start_time
        log_progresso(
            f"{log_dir}/trainning.log", 
            f'Epoch {epoch}/{num_epochs}, g_loss: {g_loss.data}, d_loss: {d_loss.data}, \
                FID: {fid_score}, Time: {epoch_duration:.2f} seconds'
        )

        losses_g.append(g_loss.data.cpu())
        losses_d.append(d_loss.data.cpu())

        vutils.save_image(
            generator(fixed_noise).data,
            os.path.join(sample_dir, f"fake_samples_epoch_{epoch:06}.jpeg"),
            normalize=True
        )

        save_checkpoint(
            epoch, generator, discriminator, optim_g, optim_d,
            losses_g, losses_d, f"{weights_path}/checkpoint.pth"
        )

        if save_model_at is not None and epoch % save_model_at == 0:
            save_checkpoint(
                epoch, generator, discriminator, optim_g, optim_d,
                losses_g, losses_d, f"{weights_path}/checkpoint_epoch_{epoch}.pth"
            )

    return losses_g, losses_d
