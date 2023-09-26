import datetime
import time

import numpy as np
import torch
import torchvision.utils as vutils

from torch.autograd import Variable, grad
from tqdm import tqdm


def log_progresso(log_file, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"

    with open(log_file, "a") as file:
        file.write(log_entry)


def save_checkpoint(epoch, generator, discriminator, optim_g, optim_d, losses_g, losses_d, path="checkpoint.pth"):
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
        losses_g=[], 
        losses_d=[]
    ):

    fixed_noise = Variable(torch.randn(sample_size, z_dim, 1, 1)).to(device)

    for epoch in range(last_epoch, num_epochs + 1):
        start_time = time.time()

        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, data in pbar:
            images, _ = data
            images = Variable(images).to(device)

            # Discriminator update
            z = Variable(torch.randn(images.size(0), z_dim, 1, 1)).to(device)
            fake_images = generator(z)

            real_loss = torch.mean(torch.abs(discriminator(images) - images))
            fake_loss = torch.mean(torch.abs(discriminator(fake_images.detach()) - fake_images))

            d_loss = real_loss - k * fake_loss

            discriminator.zero_grad()
            d_loss.backward()
            optim_d.step()

            # Generator update
            z = Variable(torch.randn(images.size(0), z_dim, 1, 1)).to(device)
            fake_images = generator(z)

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

            pbar.set_description(f'Epoch {epoch}/{num_epochs}, g_loss: {g_loss.data}, d_loss: {d_loss.data}')

        end_time = time.time()
        epoch_duration = end_time - start_time
        log_progresso(f"{log_dir}/trainning.log", f'Epoch {epoch}/{num_epochs}, g_loss: {g_loss.data}, d_loss: {d_loss.data}, FID: {fid_score}, Time: {epoch_duration:.2f} seconds')

        losses_g.append(g_loss.data.cpu())
        losses_d.append(d_loss.data.cpu())

        vutils.save_image(generator(fixed_noise).data, sample_dir + '/fake_samples_epoch_%06d.jpeg' % (epoch), normalize=True)
        save_checkpoint(epoch, generator, discriminator, optim_g, optim_d, losses_g, losses_d, f"{weights_path}/checkpoint.pth")

        if epoch % save_model_at == 0:
            save_checkpoint(epoch, generator, discriminator, optim_g, optim_d, losses_g, losses_d, f"{weights_path}/checkpoint_epoch_{epoch}.pth")

    return losses_g, losses_d
