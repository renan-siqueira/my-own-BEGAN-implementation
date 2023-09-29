"""
This script is responsible for generating and saving interpolated videos using a 
pre-trained Generative Adversarial Network (GAN) model.

Functions:
- `slerp`: Perform spherical linear interpolation (slerp) between two vectors.
- `interpolate`: Interpolate between two latent vectors using the slerp function.
- `multi_interpolate`: Generate multiple interpolated images between consecutive latent 
vectors using a trained generator model.
- `main`: Main function that loads a pre-trained generator model, creates interpolated 
images, and compiles them into a video.

Utilizes auxiliary modules for image resizing, GPU availability check, and generator 
definitions.

Dependencies:
- os: For path and directory manipulation.
- torch: For deep learning operations and tensor manipulations.
- numpy: For mathematical operations.
- cv2: For video processing and saving.
- tqdm: To display a progress bar.
- src.app.generator: Generator model definitions.
- src.app.utils: Utility functions.
- .resize_image: Functions to resize images.

Notes:
- Interpolation between latent vectors is done using spherical linear interpolation.
- The generator model is loaded from a checkpoint, placed in evaluation mode, and then 
used for generating the interpolated images.
- The generated images are compiled into a video and saved to the specified path.
"""
import os

import torch
import numpy as np
import cv2
from tqdm import tqdm

from src.app.generator import Generator
from src.app.utils import check_if_gpu_available
from .resize_image import process_and_resize_image


def slerp(val, low, high):
    """
    Perform spherical linear interpolation (slerp) between two vectors.
    
    Parameters:
    - val (float): The interpolation factor.
    - low (torch.Tensor): The starting vector.
    - high (torch.Tensor): The ending vector.
    
    Returns:
    - torch.Tensor: The interpolated vector.
    """
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos(torch.clamp(torch.matmul(low_norm, high_norm.t()), -1, 1))
    s_o = torch.sin(omega)
    if s_o == 0:
        return (1.0 - val) * low + val * high
    interpolation = (torch.sin((1.0 - val) * omega) / s_o * low)\
         + (torch.sin(val * omega) / s_o * high)
    return interpolation


def interpolate(z_1, z_2, alpha):
    """
    Interpolate between two latent vectors using slerp.
    
    Parameters:
    - z1 (torch.Tensor): The starting latent vector.
    - z2 (torch.Tensor): The ending latent vector.
    - alpha (float): The interpolation factor.
    
    Returns:
    - torch.Tensor: The interpolated latent vector.
    """
    return slerp(alpha, z_1, z_2)


def multi_interpolate(generator, z_list, steps_between):
    """
    Generate multiple interpolated images using a list of latent vectors 
    and a generator model.
    
    Parameters:
    - generator (Generator): The generator model used to produce the images.
    - z_list (list of torch.Tensor): List of latent vectors to interpolate 
    between.
    - steps_between (int): Number of interpolation steps between consecutive 
    latent vectors.
    
    Returns:
    - list of torch.Tensor: List of generated interpolated images.
    """
    generated_images = []
    for i in range(len(z_list) - 1):
        z_1 = z_list[i]
        z_2 = z_list[i + 1]
        alphas = np.linspace(0, 1, steps_between)
        for alpha in alphas:
            z_interp = interpolate(z_1, z_2, alpha)
            z_interp = z_interp.view(z_interp.size(0), z_interp.size(1), 1, 1)
            with torch.no_grad():
                generated_image = generator(z_interp)
            generated_images.append(generated_image)
    return generated_images


def main(params, path_data, path_videos_generated):
    """
    Main function to generate an interpolated video using a trained generator model.
    
    Parameters:
    - params (dict): Dictionary of parameters including number of points, steps between, fps, etc.
    - path_data (str): Path to the trained generator model's weights.
    - path_videos_generated (str): Path to save the generated video.
    """
    output_directory = os.path.join(path_videos_generated, params['training_version'])

    check_if_gpu_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_path = f'{path_data}/{params["training_version"]}/weights/checkpoint.pth'
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    generator = Generator(
        params["z_dim"], params["channels_img"], params["features_g"], img_size=params['image_size']
    ).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    z_points = [
        torch.randn(1, params["z_dim"]).to(device) for _ in range(params['interpolate_points'])
    ]

    print("Generating interpolated images...")
    generated_images = multi_interpolate(generator, z_points, params['steps_between'])

    os.makedirs(output_directory, exist_ok=True)

    upscale_width = params.get('upscale')

    if upscale_width:
        frame_size = (upscale_width, upscale_width)
    else:
        frame_size = (params["image_size"], params["image_size"])

    out = cv2.VideoWriter( # pylint: disable=no-member
        os.path.join(
            output_directory, f'video_{frame_size[0]}x{frame_size[1]}_{params["fps"]}fps.mp4'
        ),
        cv2.VideoWriter_fourcc(*'mp4v'), # pylint: disable=no-member
        params["fps"],
        frame_size
    )

    print("Writing images to video...")
    for image in tqdm(generated_images):
        image_np = image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        image_np = (image_np + 1) / 2
        frame = (image_np * 255).astype(np.uint8)

        if upscale_width:
            frame = process_and_resize_image(frame, new_width=upscale_width)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # pylint: disable=no-member
        out.write(frame)

    out.release()
