"""
This script is responsible for the main orchestration of training a Generative 
Adversarial Network (GAN) model.

It encompasses the following functionalities:
- Sets up the training environment by checking GPU availability and seeding 
  random number generators for reproducibility.
- Defines the generator and discriminator architectures and initializes their weights.
- Loads datasets and configures data loaders.
- Defines the optimizers for both the generator and discriminator.
- Handles resuming from previous training checkpoints if specified.
- Initiates the model training process using the `train_model` function.
- Monitors and logs training progress.
- Finally, visualizes and saves the generator and discriminator loss curves 
  post-training.

Dependencies:
- time: For tracking the total execution time.
- os: For directory and path manipulations.
- torch and torchvision: For deep learning operations and model architectures.
- src.app.generator: Definitions for the GAN generator.
- src.app.discriminator: Definitions for the GAN discriminator.
- src.app.training: Contains the `train_model` function that handles the training loop.
- src.app.utils: Utility functions for miscellaneous operations.

Functions:
- `main`: The main function that orchestrates the entire training process, from 
  model instantiation to the final visualization of the loss curves.

Notes:
- Uses a variety of hyperparameters and configurations passed in the `params` 
  dictionary.
- Allows for resuming training from previous checkpoints.
- Can utilize the Inception V3 model for the Frechet Inception Distance (FID) 
  metric when the image size is 128x128.
- The training version and associated data are organized into a dedicated 
  directory structure for better management.
"""
import time
import os

import torch
from torch import optim
from torchvision import models

from src.app.generator import Generator
from src.app.discriminator import Discriminator
from src.app.training import train_model
from src.app import utils


def main(params, path_data, path_dataset, path_params):
    """
    Orchestrates the training process of a Generative Adversarial Network (GAN) model.

    The function sets up the training environment, including GPU checks and seeding for 
    reproducibility. It defines the model architectures, configures data loaders, and 
    defines the optimizers for training. Depending on the configuration, training can 
    resume from a previous checkpoint. It also has provisions for the Frechet Inception 
    Distance (FID) metric for 128x128 images. After training, it logs and visualizes the 
    loss curves.

    Parameters:
    - params (dict): Dictionary containing various training parameters and configurations 
        such as batch size, learning rates, epochs, and more.
    - path_data (str): Directory path where training-related data (checkpoints, logs, samples) 
        will be stored.
    - path_dataset (str): Directory path of the dataset used for training.
    - path_params (str): Path to the configuration file containing `params`.

    Notes:
    - The training process utilizes various utility functions from the `utils` module for 
        operations like checkpoint loading, image generation, and loss plotting.
    - Depending on the image size, the function might use the Inception V3 model to 
        calculate the FID metric.
    - Outputs relevant logs and status messages throughout the training process.
    """
    time_start = time.time()
    utils.print_datetime()

    utils.check_if_gpu_available()
    utils.check_if_set_seed(params["seed"])

    print(f'Image size: {params["image_size"]}x{params["image_size"]}\n')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inception_model = None
    if params['image_size'] == 128:
        # Frechet Inception Distance (FID)
        inception_model = models.inception_v3(
            weights='Inception_V3_Weights.DEFAULT', transform_input=False, init_weights=False
        ).to(device)
        inception_model = inception_model.eval()

    generator = Generator(
        params["z_dim"], params["channels_img"], params["features_g"], img_size=params['image_size']
    ).to(device)
    generator.apply(utils.weights_init)

    discriminator = Discriminator(
        params["channels_img"], params["features_d"], img_size=params['image_size']
    ).to(device)
    discriminator.apply(utils.weights_init)

    data_loader = utils.dataloader(path_dataset, params["image_size"], params["batch_size"])

    optim_g = optim.Adam(
        generator.parameters(),
        lr=params["lr_g"],
        betas=(params['g_beta_min'], params['g_beta_max'])
    )

    optim_d = optim.Adam(
        discriminator.parameters(),
        lr=params["lr_d"],
        betas=(params['d_beta_min'], params['d_beta_max'])
    )

    resume_training = params.get('resume_training')

    if resume_training:
        training_version = params.get('training_version')
    else:
        training_version = utils.create_next_version_directory(path_data)

    data_dir = os.path.join(path_data, training_version)
    print('Training version:', training_version)

    # Create a copy of parameters in training version folder
    utils.safe_copy(params, os.path.join(data_dir, os.path.basename(path_params)))

    last_epoch, losses_g, losses_d = utils.load_checkpoint(
        os.path.join(data_dir, 'weights', 'checkpoint.pth'),
        generator, discriminator, optim_g, optim_d
    )

    losses_g, losses_d = train_model(
        inception_model=inception_model,
        generator=generator,
        discriminator=discriminator,
        weights_path= os.path.join(data_dir, 'weights'),
        sample_size=params["sample_size"],
        sample_dir= os.path.join(data_dir, 'samples'),
        optim_g=optim_g,
        optim_d=optim_d,
        k=params['k'],
        lambda_k=params['lambda_k'],
        gamma=params['gamma'],
        data_loader=data_loader,
        device=device,
        z_dim=params["z_dim"],
        num_epochs=params["num_epochs"],
        last_epoch=last_epoch,
        save_model_at=params['save_model_at'],
        log_dir = os.path.join(data_dir, 'log'),
        losses_g=losses_g,
        losses_d=losses_d,
    )

    time_end = time.time()
    time_total = (time_end - time_start) / 60

    print(f"The code took {round(time_total, 1)} minutes to execute.")
    utils.print_datetime()

    utils.plot_losses(
        losses_g, losses_d, save_plot_image=os.path.join(data_dir, f"{training_version}.jpg")
    )
