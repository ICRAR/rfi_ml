# -*- coding: utf-8 -*-
#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#

"""
GAN model that can accept a data from a single polarisation and single frequency (1D array of samples of N width per batch).
Has option to use FFT of samples (2N width per batch)
"""

from torch import nn
import torch


class Discriminator(nn.Sequential):
    """
    Determines whether the provided input is actually RFI noise
    """

    def __init__(self, width):
        """
        Construct the discriminator
        :param width: Number of samples put through the network per batch.
        """
        def layer(in_size, out_size):
            return [
                nn.Linear(in_size, out_size),
                nn.ELU(alpha=0.3),
                nn.BatchNorm1d(out_size),
                nn.Dropout(p=0.4)
            ]

        super(Discriminator, self).__init__(
            *layer(width, width // 2),
            *layer(width // 2, width // 4),
            *layer(width // 4, width // 8),
            *layer(width // 8, width // 16),
            nn.Linear(width // 16, 2),
            nn.ELU(alpha=0.3),
            nn.Softmax(dim=1)
        )


class Generator(nn.Sequential):
    """
    Generator autoencoder that will receive an array of gaussian noise, and will convert it into RFI noise.
    """

    def __init__(self, width):
        """
        Construct the generator
        :param width: Number of samples to put through the network per batch.
        :param noise_width: Width of the input noise vector to the network.
        """
        super(Generator, self).__init__()

        def layer(in_size, out_size, final=False):
            layers = [
                nn.Linear(in_size, out_size)
            ]

            if not final:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))

            return layers

        def encoder(width):
            return nn.Sequential(
                *layer(width, width),
                *layer(width, width),
                *layer(width, width)
            )
        def decoder(width):
            return nn.Sequential(
                *layer(width, width),
                *layer(width, width),
                *layer(width, width, final=True),
            )

        self.encoder = encoder(width)
        self.decoder = decoder(width)

        self.width = width
        self.is_autoencoder = False

        def init_weights(m):
            if type(m) is nn.Linear:
                nn.init.xavier_uniform(m.weight)

        self.apply(init_weights)

    def forward(self, x):
        if self.is_autoencoder:
            return self.decoder(self.encoder(x))
        else:
            return self.decoder(x)

    def get_noise_width(self):
        """
        :return: Width of the noise vector that the generator expects when in decoder mode
        """
        return self.width // 16

    def set_autoencoder(self, is_autoencoder):
        self.is_autoencoder = is_autoencoder
        return self