"""
Module for defining the Discriminator of a Generative Adversarial Network (GAN).

This module introduces a discriminator architecture designed with an encoder-decoder structure. 
The Discriminator aims to distinguish between real and generated images. The encoding section 
downsamples the input image, extracting important features. The decoding section then upsamples 
the features back into an image space. Each block in the discriminator consists of a convolution 
or transposed convolution followed by batch normalization and a leaky ReLU activation.

Classes:
- `Discriminator`: The main class representing the discriminator model. It constructs the 
                   discriminator architecture using the provided parameters and defines the 
                   forward method for processing input images through the model.

Dependencies:
- torch.nn: For neural network operations and building block definitions.
- math: For mathematical operations, particularly for dynamically determining model depth.

Notes:
- This discriminator is designed to be used with a corresponding generator in a GAN setup.
- The encoder-decoder structure can be tailored based on specific architectural needs and 
  the desired input image size.
"""
import math
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Represents the Discriminator of a Generative Adversarial Network (GAN).
    
    The Discriminator maps input images through an encoder and decoder sequence to produce 
    output images. The aim is to discriminate between real and fake images.
    
    Attributes:
    - encoder (nn.Sequential): Encoder part of the model.
    - decoder (nn.Sequential): Decoder part of the model.
    
    Methods:
    - forward(x): Processes the input image `x` through the Discriminator.
    - _block(...): Returns a block of layers used in the Discriminator's architecture.
    - _make_layers(...): Creates and returns a sequence of blocks for encoding or decoding.
    """
    def __init__(self, channels_img, features_d, img_size):
        """
        Initializes the Discriminator with the given parameters.
        
        Parameters:
        - channels_img (int): Number of channels in the input image.
        - features_d (int): Base number of features in the discriminator.
        - img_size (int): Size of the input image (assumed to be square).
        """
        super(Discriminator, self).__init__()

        self.encoder = self._make_layers(channels_img, features_d, img_size, mode='encoder')
        self.decoder = self._make_layers(channels_img, features_d, img_size, mode='decoder')

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, mode='encoder'):
        """
        Returns a block of layers used either in the encoding or decoding parts of the 
        Discriminator.
        
        Parameters:
        - in_channels (int): Number of input channels to the block.
        - out_channels (int): Number of output channels from the block.
        - kernel_size (int): Size of the convolutional kernel.
        - stride (int): Stride of the convolutional kernel.
        - padding (int): Padding added to the input before convolution.
        - mode (str): Either 'encoder' or 'decoder' to determine the type of block.

        Returns:
        - nn.Sequential: A block for the Discriminator model.
        """
        layers = []
        if mode == 'encoder':
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.25))
        elif mode == 'decoder':
            layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _make_layers(self, channels_img, features_d, img_size, mode):
        """
        Creates a sequence of blocks for either encoding or decoding.

        Parameters:
        - channels_img (int): Number of channels in the input image.
        - features_d (int): Base number of features in the discriminator.
        - img_size (int): Size of the input image (assumed to be square).
        - mode (str): Either 'encoder' or 'decoder' to determine the type of sequence.

        Returns:
        - nn.Sequential: A sequence of blocks.
        """
        layers = []
        in_channels = channels_img
        out_channels = features_d
        num_blocks = int(math.log2(img_size)) - 2

        if mode == 'encoder':
            layers.append(self._block(in_channels, out_channels, 4, 2, 1, mode))
            in_channels = out_channels
            for _ in range(num_blocks - 1):
                out_channels *= 2
                layers.append(self._block(in_channels, out_channels, 4, 2, 1, mode))
                in_channels = out_channels
        elif mode == 'decoder':
            in_channels = features_d * (2 ** (num_blocks - 1))
            out_channels = in_channels // 2
            for _ in range(num_blocks - 1):
                layers.append(self._block(in_channels, out_channels, 4, 2, 1, mode))
                in_channels = out_channels
                out_channels //= 2
            layers.append(self._block(in_channels, channels_img, 4, 2, 1, mode))

        return nn.Sequential(*layers)

    def forward(self, x_vector):
        """
        Processes the input image through the Discriminator.

        Parameters:
        - x_vector (torch.Tensor): Input image tensor.

        Returns:
        - torch.Tensor: Output image tensor after passing through the Discriminator.
        """
        x_vector = self.encoder(x_vector)
        x_vector = self.decoder(x_vector)
        return x_vector
