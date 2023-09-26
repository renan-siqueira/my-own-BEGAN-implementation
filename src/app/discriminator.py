import math
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, img_size):
        super(Discriminator, self).__init__()

        self.encoder = self._make_layers(channels_img, features_d, img_size, mode='encoder')
        self.decoder = self._make_layers(channels_img, features_d, img_size, mode='decoder')

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, mode='encoder'):
        layers = []
        if mode == 'encoder':
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif mode == 'decoder':
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)

    def _make_layers(self, channels_img, features_d, img_size, mode):
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
