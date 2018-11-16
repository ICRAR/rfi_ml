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

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Sequential):
    """
    Determines whether the provided input is actually RFI noise
    """

    def __init__(self, width):
        """
        Construct the discriminator
        :param width: Number of samples put through the network per batch.
        """

        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=128)
        self.max_pool1 = nn.MaxPool1d(2, stride=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=64)
        self.max_pool2 = nn.MaxPool1d(2, stride=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=32)
        self.max_pool3 = nn.MaxPool1d(2, stride=2)

        self.fc1 = nn.Linear(53248, 13312)
        self.batch_norm1 = nn.BatchNorm1d(13312)

        self.fc2 = nn.Linear(13312, 3328)
        self.batch_norm2 = nn.BatchNorm1d(3328)

        self.fc3 = nn.Linear(3328, width // 4)     # output size = 512
        self.batch_norm3 = nn.BatchNorm1d(width // 4)

        self.fc4 = nn.Linear(width // 4, width // 8) # output size = 256
        self.batch_norm4 = nn.BatchNorm1d(width // 8)

        self.fc5 = nn.Linear(width // 8, width // 16)
        self.batch_norm5 = nn.BatchNorm1d(width//16)

        self.fc6 = nn.Linear(width // 16, 2)

    def forward(self, x):
        x = self.max_pool1(F.elu(self.conv1(x), alpha=0.3))
        x = self.max_pool2(F.elu(self.conv2(x), alpha=0.3))
        x = self.max_pool3(F.elu(self.conv3(x), alpha=0.3))
        x = x.view(-1, x.size()[1]*x.size()[2])

        x = self.batch_norm1(F.elu(self.fc1(x), alpha=0.3))
        x = self.batch_norm2(F.elu(self.fc2(x), alpha=0.3))
        x = self.batch_norm3(F.elu(self.fc3(x), alpha=0.3))
        x = self.batch_norm4(F.elu(self.fc4(x), alpha=0.3))
        x = self.batch_norm5(F.elu(self.fc5(x), alpha=0.3))
        # todo: try tanh
        x = F.elu(self.fc6(x), alpha=0.3)
        # todo: check this softmax
        x = F.tanh(x, dim=1)
        return x

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

        def layer(in_size, out_size, dropout=True):
            layers = [
                nn.Linear(in_size, out_size)
            ]

            if dropout:
                layers.append(nn.Dropout(0.5))

            return layers

        def encoder(width):
            return nn.Sequential(
                *layer(width, width),
                *layer(width, width, dropout=False),
            )
        def decoder(width):
            return nn.Sequential(
                *layer(width, width),
                *layer(width, width, dropout=False),
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