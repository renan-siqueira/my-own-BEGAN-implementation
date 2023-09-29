"""
Utilities for training and evaluation in a deep learning context.

This module provides utility functions for various tasks such as directory management, 
checking GPU availability, data loading, weight initialization, checkpoint management, 
and visualization.

Imports:
- os, datetime, json: Standard libraries for system operations and data handling.
- torch, matplotlib, torchvision: Libraries for deep learning operations and data visualization.

Functions:
- create_next_version_directory: Creates a new versioned directory for model runs.
- print_datetime: Prints the current date and time.
- check_if_gpu_available: Checks and prints if GPU is available and lists the available GPUs.
- check_if_set_seed: Sets a seed for reproducibility.
- dataloader: Creates a data loader from images in a directory.
- weights_init: Initializes weights for neural networks.
- load_checkpoint: Loads model and optimizer states from a checkpoint.
- plot_losses: Visualizes training losses for generator and discriminator.
- safe_copy: Safely saves JSON data to a given path, preventing overwrites.
"""
import os
import datetime
import json

import torch
from torch import nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


def create_next_version_directory(base_dir):
    """
    Create and return the next versioned directory based on existing version directories.
    
    Parameters:
    - base_dir (str): The base directory where version directories are located.
    
    Returns:
    - str: The name of the created directory (e.g., "v3").
    """
    versions = [
        d for d in os.listdir(base_dir) if d.startswith('v') and os.path.isdir(
            os.path.join(base_dir, d)
        )
    ]

    if not versions:
        next_version = 1
    else:
        next_version = max(int(v[1:]) for v in versions) + 1

    new_dir_base = os.path.join(base_dir, f'v{next_version}')

    for sub_dir in ['', 'samples', 'weights', 'log']:
        os.makedirs(os.path.join(new_dir_base, sub_dir), exist_ok=True)

    return f"v{next_version}"


def print_datetime(label="Current Date and Time"):
    """
    Print the current date and time with a given label.
    
    Parameters:
    - label (str): A descriptive label to print before the date and time.
    """
    data_hora_atual = datetime.datetime.now()
    data_hora_formatada = data_hora_atual.strftime("%d/%m/%Y %H:%M:%S")
    print(f'\n{label}: {data_hora_formatada}')


def check_if_gpu_available():
    """
    Checks if a GPU is available and prints the GPU details.
    """
    print('Use GPU:', torch.cuda.is_available())

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPUs available: {device_count}")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {device_name}")
    else:
        print("No GPU available.")


def check_if_set_seed(seed=None):
    """
    Sets a seed for reproducibility in PyTorch operations.
    
    Parameters:
    - seed (int, optional): The seed to set. If not provided, indicates using a random seed.
    """
    if seed:
        torch.manual_seed(seed)
        print(f'Using the Seed: {seed}')
    else:
        print('Using random seed.')


def dataloader(directory, image_size, batch_size):
    """
    Creates a PyTorch data loader for images from a given directory.
    
    Parameters:
    - directory (str): The path to the image dataset directory.
    - image_size (int): The target size for resizing the images.
    - batch_size (int): The number of samples per batch.
    
    Returns:
    - DataLoader: A PyTorch DataLoader object for the image dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(directory, transform=transform)
    loader = torch.utils.data.DataLoader( # type: ignore
        dataset, batch_size=batch_size, shuffle=True
    )
    return loader


def weights_init(module):
    """
    Initializes the weights of a PyTorch model.
    
    Parameters:
    - module (nn.Module): A PyTorch module whose weights will be initialized.
    """
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def load_checkpoint(path, generator, discriminator, optim_g, optim_d):
    """
    Loads a model and optimizer states from a checkpoint.
    
    Parameters:
    - path (str): The path to the checkpoint file.
    - generator (nn.Module): The generator model.
    - discriminator (nn.Module): The discriminator model.
    - optim_g (Optimizer): The optimizer for the generator.
    - optim_d (Optimizer): The optimizer for the discriminator.
    
    Returns:
    - tuple: Starting epoch number, generator losses, discriminator losses.
    """
    if not os.path.exists(path):
        print("No checkpoint found.")
        return 1, [], []

    checkpoint = torch.load(path)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optim_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optim_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    epoch = checkpoint['epoch'] + 1
    losses_g = checkpoint['losses_g']
    losses_d = checkpoint['losses_d']

    print(f'Checkpoint loaded, starting from epoch {epoch}')
    return epoch, losses_g, losses_d


def plot_losses(losses_g, losses_d, save_plot_image):
    """
    Plots the training losses for the generator and discriminator.
    
    Parameters:
    - losses_g (list): List of generator loss values.
    - losses_d (list): List of discriminator loss values.
    - save_plot_image (str): Path where the plotted image will be saved.
    """
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(losses_g, label="G")
    plt.plot(losses_d, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_plot_image, bbox_inches='tight')
    plt.show()


def safe_copy(json_data, dest_path):
    """
    Safely saves JSON data to a given path, preventing overwrites by adding a 
    version number.
    
    Parameters:
    - json_data (dict): The JSON data to be saved.
    - dest_path (str): The target path to save the JSON data.
    
    Returns:
    - str: The path where the JSON data was saved.
    """
    dest_dir, filename = os.path.split(dest_path)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if os.path.exists(dest_path):
        base_name, ext = os.path.splitext(filename)
        counter = 1

        while os.path.exists(os.path.join(dest_dir, f"{base_name}_{counter}{ext}")):
            counter += 1

        dest_path = os.path.join(dest_dir, f"{base_name}_{counter}{ext}")

    with open(dest_path, 'w', encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

    return dest_path
