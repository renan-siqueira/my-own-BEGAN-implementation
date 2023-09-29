import json
import argparse
import os

from src.config import settings
from src.modules import run_training, image_generate, video_generate


def get_params(path_file):
    with open(path_file, 'r') as f:
        params = json.load(f)
    
    return params


def args_to_dict(args):
    return {k.replace("-", "_"): v for k, v in vars(args).items() if v is not None}


def update_params_from_args(params, arg_dict):
    for key, value in arg_dict.items():
        if key in params:
            params[key] = value

def main(args):

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
    parser = argparse.ArgumentParser(description="Script to train the generator, generate images and/or videos from a trained generator")

    # feature args
    parser.add_argument('--image', action='store_true', help='If true, generates images')
    parser.add_argument('--training', action='store_true', help='If true, executes the training')
    parser.add_argument('--video', action='store_true', help='If true, generates videos')

    # Exclusive args of --training
    parser.add_argument('--batch-size', type=int, help='[--training] Sets the batch size.')
    parser.add_argument('--channels-img', type=int, help='[--training] Sets the number of image channels.')
    parser.add_argument('--d-beta-max', type=float, help='[--training] Sets the d_beta_max value.')
    parser.add_argument('--d-beta-min', type=float, help='[--training] Sets the d_beta_min value.')
    parser.add_argument('--features-d', type=int, help='[--training] Sets the features for discriminator.')
    parser.add_argument('--features-g', type=int, help='[--training] Sets the features for generator.')
    parser.add_argument('--gamma', type=float, help='[--training] Sets the gamma value.')
    parser.add_argument('--g-beta-max', type=float, help='[--training] Sets the g_beta_max value.')
    parser.add_argument('--g-beta-min', type=float, help='[--training] Sets the g_beta_min value.')
    parser.add_argument('--image-size', type=int, help='[--training] Sets the size of the images to be generated.')
    parser.add_argument('--k', type=float, help='[--training] Sets the initial value for equilibrium.')
    parser.add_argument('--lambda-k', type=float, help='[--training] Sets the lambda_k value.')
    parser.add_argument('--lr-d', type=float, help='[--training] Sets the learning rate for discriminator.')
    parser.add_argument('--lr-g', type=float, help='[--training] Sets the learning rate for generator.')
    parser.add_argument('--num-epochs', type=int, help='[--training] Sets the number of training epochs.')
    parser.add_argument('--resume-training', action='store_true', help='[--training] If true, resumes the specific training from checkpoint.')
    parser.add_argument('--save-model-at', type=int, help='[--training] Sets the frequency (in epochs) to save the model (checkpoint).')
    parser.add_argument('--z-dim', type=int, help='[--training] Sets the dimensionality of the latent space.')

    # Exclusive args of --image
    parser.add_argument('--num-samples', type=int, help='[--image] Sets how many images you want to generate.')

    # Exclusive args of --video
    parser.add_argument('--interpolate-points', type=int, help='[--video] Sets the number of interpolation points.')
    parser.add_argument('--fps', type=int, help='[--video] Sets the number of frames per second of the video.')
    parser.add_argument('--steps-between', type=int, help='[--video] Sets the number os images to be generated between each interpolated point.')

    # Other args
    parser.add_argument('--seed', type=int, help='[--training|--image |--video] Sets the random seed for reproducibility.')
    parser.add_argument('--training-version', type=str, help='[--training|--image |--video] Sets the training version to use.')
    parser.add_argument('--upscale', type=int, help='[--image|--video] Sets the upscale width. Can be None or an integer value.')
    
    args = parser.parse_args()

    main(args)
