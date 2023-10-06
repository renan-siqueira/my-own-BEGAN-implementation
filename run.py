"""
Module for training a Generative Adversarial Network (GAN), and for generating images and videos 
using the trained model.

This module provides utilities for parsing command line arguments, extracting model parameters 
from JSON, updating model parameters 
with parsed arguments, and executing training or generation processes based on the given command 
line inputs.

Functions:
- `get_params`: Load parameters from a given JSON file.
- `args_to_dict`: Convert the parsed arguments to a dictionary format.
- `update_params_from_args`: Update model parameters with values provided as command-line arguments.
- `main`: The main function to run training, image generation, or video generation based on the 
provided arguments.

Dependencies:
- json: For reading model parameters from JSON files.
- argparse: For parsing command line arguments.
- src.config: Module containing configuration settings and default paths.
- src.modules: Collection of functions for training, image generation, and video generation.

Notes:
- This module acts as a CLI tool, allowing users to train models and generate outputs via terminal 
commands.
- The specific features and configurations for the training, image, and video generation processes 
are determined by command line arguments and JSON files.

Usage:
Run this module from the command line, and provide necessary arguments. 

For example: 

$ python run.py --training --batch-size 64 --num-epochs 100
$ python run.py --training --batch-size 64 --num-epochs 250 --resume-training --training-version v1

$ python run.py --image --training-version v1 
$ python run.py --image --training-version v1 --upscale 128

$ python run.py --video --training-version v1
$ python run.py --video --training-version v1 --upscale 128

This will initiate the training with a batch size of 64 for 100 epochs.
"""
import json
import argparse

from src.config import settings
from src.modules import run_training, image_generate, video_generate


def get_params(path_file):
    """
    Retrieve parameters from a JSON file.

    Parameters:
    - path_file (str): Path to the JSON file containing the parameters.

    Returns:
    - dict: Dictionary containing the loaded parameters.
    """
    with open(path_file, 'r', encoding='utf-8') as file:
        params = json.load(file)

    return params


def args_to_dict(args):
    """
    Convert parsed arguments into a dictionary, replacing hyphens with underscores.

    Parameters:
    - args (Namespace): Parsed command-line arguments (from argparse).

    Returns:
    - dict: Dictionary representation of the parsed arguments.
    """
    return {k.replace("-", "_"): v for k, v in vars(args).items() if v is not None}


def update_params_from_args(params, arg_dict):
    """
    Update a parameters dictionary with values provided in another dictionary.

    Parameters:
    - params (dict): The original dictionary to be updated.
    - arg_dict (dict): Dictionary containing new values to update `params`.

    Returns:
    None.

    Notes:
    - Modifies the `params` dictionary in-place.
    """
    for key, value in arg_dict.items():
        if key in params:
            params[key] = value


def main(args):
    """
    Main function to run training, image generation, or video generation based on provided 
    arguments.

    Parameters:
    - args (Namespace): Parsed command-line arguments (from argparse).

    Returns:
    None.

    Notes:
    - Determines which process (training, image generation, video generation) to run based on the 
      flags provided in `args`.
    """
    arg_dict = args_to_dict(args)
    params = get_params(settings.PATH_PARAMS)
    update_params_from_args(params, arg_dict)

    if args.training:
        run_training.main(params, settings.PATH_DATA, settings.PATH_DATASET, settings.PATH_PARAMS)

    if args.image:
        image_generate.main(params, settings.PATH_DATA, settings.PATH_IMAGES_GENERATED)

    if args.video:
        video_generate.main(params, settings.PATH_DATA, settings.PATH_VIDEOS_GENERATED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to train the generator, generate images or videos from trained model"
    )

    # feature args
    parser.add_argument('--image', action='store_true', help='If true, generates images')
    parser.add_argument('--training', action='store_true', help='If true, executes the training')
    parser.add_argument('--video', action='store_true', help='If true, generates videos')

    # Exclusive args of --training
    parser.add_argument('--batch-size', type=int, help='[--training] Sets the batch size.')
    parser.add_argument(
        '--channels-img', type=int, help='[--training] Sets the number of image channels.'
    )
    parser.add_argument('--d-beta-max', type=float, help='[--training] Sets the d_beta_max value.')
    parser.add_argument('--d-beta-min', type=float, help='[--training] Sets the d_beta_min value.')
    parser.add_argument(
        '--features-d', type=int, help='[--training] Sets the features for discriminator.'
    )
    parser.add_argument(
        '--features-g', type=int, help='[--training] Sets the features for generator.'
    )
    parser.add_argument('--gamma', type=float, help='[--training] Sets the gamma value.')
    parser.add_argument('--g-beta-max', type=float, help='[--training] Sets the g_beta_max value.')
    parser.add_argument('--g-beta-min', type=float, help='[--training] Sets the g_beta_min value.')
    parser.add_argument(
        '--image-size', type=int, help='[--training] Sets the size of the images to be generated.'
    )
    parser.add_argument(
        '--k', type=float, help='[--training] Sets the initial value for equilibrium.'
    )
    parser.add_argument('--lambda-k', type=float, help='[--training] Sets the lambda_k value.')
    parser.add_argument(
        '--lr-d', type=float, help='[--training] Sets the learning rate for discriminator.'
    )
    parser.add_argument(
        '--lr-g', type=float, help='[--training] Sets the learning rate for generator.'
    )
    parser.add_argument(
        '--num-epochs', type=int, help='[--training] Sets the number of training epochs.'
    )
    parser.add_argument(
        '--resume-training',
        action='store_true',
        help='[--training] If true, resumes the specific training from checkpoint.'
    )
    parser.add_argument(
        '--save-model-at',
        type=int,
        help='[--training] Sets the frequency (in epochs) to save the model (checkpoint).'
    )
    parser.add_argument(
        '--z-dim', type=int, help='[--training] Sets the dimensionality of the latent space.'
    )

    # Exclusive args of --image
    parser.add_argument(
        '--num-samples', type=int, help='[--image] Sets how many images you want to generate.'
    )

    # Exclusive args of --video
    parser.add_argument(
        '--interpolate-points', type=int, help='[--video] Sets the number of interpolation points.'
    )
    parser.add_argument(
        '--fps', type=int, help='[--video] Sets the number of frames per second of the video.'
    )
    parser.add_argument(
        '--steps-between',
        type=int,
        help='[--video] Sets the number os images to be generated between each interpolated point.'
    )

    # Other args
    parser.add_argument(
        '--seed',
        type=int,
        help='[--training|--image |--video] Sets the random seed for reproducibility.'
    )
    parser.add_argument(
        '--training-version',
        type=str,
        help='[--training|--image |--video] Sets the training version to use.'
    )
    parser.add_argument(
        '--upscale',
        type=int,
        help='[--image|--video] Sets the upscale width. Can be None or an integer value.'
    )

    parsed_args = parser.parse_args()

    main(parsed_args)
