"""
Module for defining the Generator of a Generative Adversarial Network (GAN).

This module introduces a generator architecture using transposed convolutions, enabling the network 
to upscale a latent space vector into a 2D image. Each block in the generator consists of a 
transposed convolution followed by batch normalization and a leaky ReLU activation. 
The generator expands the latent vector's dimensions until it matches the desired image size.

Classes:
- `Generator`: The main class representing the generator model. It builds the generator architecture 
               from the provided parameters and defines the forward method for image generation.

Dependencies:
- torch.nn: For neural network operations and building block definitions.
- math: For mathematical operations.

Notes:
- This generator is designed to be used with a corresponding discriminator in a GAN setup.
- The generator architecture can be adapted based on the desired output image size and depth.
"""
import math
import torch.nn as nn

class Generator(nn.Module):
    """
    Represents the Generator of a Generative Adversarial Network (GAN) that upscales latent vectors 
    into images.

    Attributes:
    - gen (nn.Sequential): The core sequential model of the generator composed of several blocks.

    Methods:
    - forward(x): Produces an image from the given latent space vector `x`.
    - _block(in_channels, out_channels, kernel_size, stride, padding): Returns a block used in the 
      generator architecture consisting of a transposed convolution, batch normalization, and a 
      leaky ReLU.

    Note:
    - The generator architecture is defined during initialization and it's designed to upscale a 
      latent vector to match the desired image size using transposed convolutions.
    """
    def __init__(self, z_dim, channels_img, features_g, img_size):
        """
        Initializes the Generator model by setting up its architecture.

        Parameters:
        - z_dim (int): Dimension of the input latent vector.
        - channels_img (int): Number of channels in the output image.
        - features_g (int): Base number of features in the generator.
        - img_size (int): Size of the output image (assumed to be square).
        """
        super(Generator, self).__init__()

        layers = []

        num_blocks = int(math.log2(img_size)) - 3
        out_channels = features_g * (2 ** num_blocks)

        layers.append(nn.ConvTranspose2d(z_dim, out_channels, 4, 1, 0))
        layers.append(nn.ReLU(True))

        for _ in range(num_blocks):
            in_channels = out_channels
            out_channels = in_channels // 2
            layers.append(self._block(in_channels, out_channels, 4, 2, 1))

        layers.append(nn.ConvTranspose2d(out_channels, channels_img, 4, 2, 1))
        layers.append(nn.Tanh())

        self.gen = nn.Sequential(*layers)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Returns a block used in the generator architecture.

        A block consists of a transposed convolution followed by batch normalization 
        and a leaky ReLU activation.

        Parameters:
        - in_channels (int): Number of input channels to the block.
        - out_channels (int): Number of output channels from the block.
        - kernel_size (int): Size of the convolutional kernel.
        - stride (int): Stride of the convolutional kernel.
        - padding (int): Padding added to the input before convolution.

        Returns:
        - nn.Sequential: A block for the generator model.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x_vector):
        """
        Produces an image from the given latent space vector.

        Parameters:
        - x_vector (torch.Tensor): Input latent vector.

        Returns:
        - torch.Tensor: An image tensor generated from the latent vector.
        """
        return self.gen(x_vector)
