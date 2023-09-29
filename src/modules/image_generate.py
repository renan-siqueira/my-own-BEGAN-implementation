"""
This script is responsible for generating and saving images using a pre-trained 
Generative Adversarial Network (GAN) model.

Functions:
- `generate_latent_points`: Generates random latent points that serve as input for 
the generator model.
- `generate_images`: Uses the generator model to produce images from latent points.
- `tensor_to_pil_image`: Converts a PyTorch tensor into a PIL image.
- `main`: Main function that loads a pre-trained generator model, generates images, 
and saves them individually and in a grid format.

Utilizes auxiliary modules for image resizing, GPU availability check, and generator 
definitions.

Dependencies:
- os: For path and directory manipulation.
- numpy: For mathematical operations.
- torch and torchvision: For deep learning operations and image manipulation.
- PIL: For image processing.
- tqdm: To display a progress bar.
- src.app.generator: Generator model definitions.
- src.app.utils: Utility functions.
- .resize_image: Functions to resize images.

Notes:
- Generated images are normalized to the range [0, 1] and can be resized to a specified width.
- The generator model is loaded from a checkpoint and placed in evaluation mode before image 
generation.
"""
import os

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from src.app.generator import Generator
from src.app.utils import check_if_gpu_available
from .resize_image import process_and_resize_image


def generate_latent_points(latent_dimension, num_samples, device):
    """
    Generate random latent points as input for the generator model.
    
    Parameters:
    - latent_dimension (int): The size of the latent space.
    - num_samples (int): The number of latent points to generate.
    - device (torch.device): The device to which the generated points 
    should be assigned (e.g., 'cuda' or 'cpu').

    Returns:
    - torch.Tensor: A tensor of randomly generated latent points.
    """
    points = torch.randn(num_samples, latent_dimension, device=device)
    return points


def generate_images(model, latent_dimension, num_samples, device):
    """
    Generate images using a pre-trained generator model.
    
    Parameters:
    - model (torch.nn.Module): The pre-trained generator model.
    - latent_dimension (int): The size of the latent space.
    - num_samples (int): The number of images to generate.
    - device (torch.device): The device on which the model and points are located.

    Returns:
    - torch.Tensor: A tensor of generated images.
    """
    points = generate_latent_points(latent_dimension, num_samples, device)
    points = points.view(num_samples, latent_dimension, 1, 1)
    with torch.no_grad():
        images = model(points)
    return images


def tensor_to_pil_image(img_tensor):
    """
    Convert a PyTorch tensor to a PIL Image.
    
    Parameters:
    - img_tensor (torch.Tensor): A tensor representing an image.

    Returns:
    - PIL.Image: The converted image.
    """
    img_array = img_tensor.clone().detach().cpu().numpy()
    img_array = img_array.transpose(1, 2, 0)
    img_array = (img_array * 255).round().astype(np.uint8)
    return Image.fromarray(img_array)


def main(params, path_data, path_images_generated):
    """
    Main function to load a pre-trained generator model, produce images, and save them.
    
    Parameters:
    - params (dict): A dictionary of parameters containing model and generation settings.
    - path_data (str): The path to the directory containing the model checkpoint.
    - path_images_generated (str): The path to the directory where the generated images 
    will be saved.
    """
    num_samples = params['num_samples']
    output_directory = os.path.join(path_images_generated, params["training_version"])

    check_if_gpu_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_path = f'{path_data}/{params["training_version"]}/weights/checkpoint.pth'

    checkpoint = torch.load(checkpoint_path)

    generator = Generator(
        params["z_dim"], params["channels_img"], params["features_g"], img_size=params['image_size']
    ).to(device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    latent_dimension = params['z_dim']

    print("Generating images...")
    images = generate_images(generator, latent_dimension, num_samples, device)
    images = (images + 1) / 2.0

    os.makedirs(output_directory, exist_ok=True)

    upscale_width = params.get('upscale')
    image_size_str = f"{params['image_size']}x{params['image_size']}"

    print("Saving individual images...")
    for i in tqdm(range(num_samples)):
        individual_img = images[i].cpu().clamp(0, 1)
        img = tensor_to_pil_image(individual_img)

        if upscale_width:
            image_size_str = f"{upscale_width}x{upscale_width}"
            img = np.asarray(img)
            img = process_and_resize_image(img, new_width=upscale_width)
            img = Image.fromarray(img)

        img_path = os.path.join(output_directory, f'image_{image_size_str}_{i}.jpg')
        img.save(img_path)

    print("Saving image grid...")
    grid_img = vutils.make_grid(images, nrow=int(num_samples**0.5), padding=2, normalize=True)
    img_grid = tensor_to_pil_image(grid_img.cpu())

    if upscale_width:
        img_grid = np.asarray(img_grid)
        img_grid = process_and_resize_image(img_grid, new_width=upscale_width)
        img_grid = Image.fromarray(img_grid)

    img_grid_path = os.path.join(output_directory, f'grid_{image_size_str}.jpg')
    img_grid.save(img_grid_path)

    print(f"Images saved to {output_directory}")
