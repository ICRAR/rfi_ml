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

import torch.nn as nn


class Discriminator(nn.Sequential):
    """
    Determines whether the provided input is actually RFI noise
    """

    def __init__(self, config):
        """
        Construct the discriminator
        """
        def layer(in_size, out_size, final=False):
            layers = [
                nn.Linear(in_size, out_size),

            ]

            if final:
                # layers.append(nn.Softmax(dim=1))
                layers.append(nn.Sigmoid())  # 0 - 1 for single labels
            else:
                layers.append(nn.Softsign())
                layers.append(nn.Dropout(0.1))

            return layers

        width = config.WIDTH
        super(Discriminator, self).__init__(
            *layer(width, width // 2),
            *layer(width // 2, width // 4),
            *layer(width // 4, width // 8),
            *layer(width // 8, width // 16),
            *layer(width // 16, width // 32),
            *layer(width // 32, width // 64),
            *layer(width // 64, width // 128),
            *layer(width // 128, 1, final=True)
        )

class Generator(nn.Sequential):
    """
    Generator autoencoder that will receive an array of gaussian noise, and will convert it into RFI noise.

    Generator needs to be pre-trained as an autoencoder, then chopped in half and the decoder part should be used
    along with a random noise vector of inputs to the hidden layer
    """

    def __init__(self, config):
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
                layers.append(nn.Softsign())
                layers.append(nn.Dropout(0.1))

            return layers

        width = config.WIDTH
        hidden1 = int(width * 0.833)
        hidden2 = int(width * 0.666)
        hidden3 = int(width * 0.5)
        def encoder(width):
            return nn.Sequential(
                *layer(width, hidden1),
                *layer(hidden1, hidden2),
                *layer(hidden2, hidden3)
            )
        def decoder(width):
            return nn.Sequential(
                *layer(hidden3, hidden2),
                *layer(hidden2, hidden1),
                *layer(hidden1, width, final=True),
            )

        additional_input_width = int(width * 0.125)
        additional_hidden1_width = int(width * 0.25)
        self.additional_input_layer = nn.Sequential(
            *layer(additional_input_width, additional_hidden1_width),
            *layer(additional_hidden1_width, hidden3)
        )

        self.additional_output_layer = nn.Sequential(
            nn.Softsign()
        )
        self.encoder = encoder(width)
        self.decoder = decoder(width)

        self.noise_width = additional_input_width  # hidden3
        self.is_autoencoder = False

        def init_weights(m):
            if type(m) is nn.Linear:
                nn.init.xavier_uniform(m.weight)

        self.apply(init_weights)

    def forward(self, x):
        if self.is_autoencoder:
            return self.decoder(self.encoder(x))
        else:
            return self.additional_output_layer(self.decoder(self.additional_input_layer(x)))

    def get_noise_width(self):
        """
        :return: Width of the noise vector that the generator expects when in decoder mode
        """
        return self.noise_width

    def set_autoencoder(self, is_autoencoder):
        self.is_autoencoder = is_autoencoder
        return self
