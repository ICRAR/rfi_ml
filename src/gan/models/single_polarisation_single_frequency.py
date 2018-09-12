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


class Discriminator(nn.Sequential):
    """
    Determines whether the provided input is actually RFI noise
    """

    def __init__(self, width):
        """
        Construct the discriminator
        :param width: Number of samples put through the network per batch.
        """
        super(Discriminator, self).__init__(
            nn.Linear(width, width),
            nn.ELU(alpha=0.3),
            nn.BatchNorm1d(10),
            nn.Dropout(p=0.4),

            nn.Linear(width, width // 2),
            nn.ELU(alpha=0.3),
            nn.BatchNorm1d(10),
            nn.Dropout(p=0.4),

            nn.Linear(width // 2, width // 4),
            nn.ELU(alpha=0.3),
            nn.BatchNorm1d(10),
            nn.Dropout(p=0.4),

            nn.Linear(width // 4, 2),
            nn.ELU(alpha=0.3),
            nn.BatchNorm1d(10),
            nn.Dropout(p=0.4),
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
        super(Generator, self).__init__(
            nn.Linear(width // 16, width // 8),
            nn.ELU(alpha=0.3),
            nn.BatchNorm1d(10),
            nn.Dropout(p=0.4),

            nn.Linear(width // 8, width // 4),
            nn.ELU(alpha=0.3),
            nn.BatchNorm1d(10),
            nn.Dropout(p=0.4),

            nn.Linear(width // 4, width // 2),
            nn.ELU(alpha=0.3),
            nn.BatchNorm1d(10),
            nn.Dropout(p=0.4),

            nn.Linear(width // 2, width),
            nn.ELU()
        )
        self.width = width

    def get_noise_width(self):
        """
        :return: Width of the noise vector that the generator expects
        """
        return self.width // 16
