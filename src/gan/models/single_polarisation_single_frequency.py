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
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2)
        self.max_pool1 = nn.MaxPool1d(2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=2)
        self.max_pool2 = nn.MaxPool1d(2, stride=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=2)
        self.max_pool3 = nn.MaxPool1d(2, stride=2)

        self.fc1 = nn.Linear(8064, width)               # output size = 2048
        self.batch_norm1 = nn.BatchNorm1d(width)

        self.fc2 = nn.Linear(width, width // 4)       # output size = 512
        self.batch_norm2 = nn.BatchNorm1d(width // 4)

        self.fc3 = nn.Linear(width // 4, width // 8) # output size = 256
        self.batch_norm3 = nn.BatchNorm1d(width//8)

        self.fc4 = nn.Linear(width // 8, width // 16) # output size = 256
        self.batch_norm4 = nn.BatchNorm1d(width//16)

        self.fc5 = nn.Linear(width // 16, 2)

    def forward(self, x):
        x = self.max_pool1(F.hardtanh(self.conv1(x)))
        x = self.max_pool2(F.hardtanh(self.conv2(x)))
        x = self.max_pool3(F.hardtanh(self.conv3(x)))
        x = x.view(-1, x.size()[1]*x.size()[2])
        #x = F.dropout(x)
        x = self.batch_norm1(F.hardtanh(self.fc1(x)))
        x = self.batch_norm2(F.hardtanh(self.fc2(x)))
        x = self.batch_norm3(F.hardtanh(self.fc3(x)))
        x = self.batch_norm4(F.hardtanh(self.fc4(x)))
        # todo: try tanh
        x = F.hardtanh(self.fc5(x))
        # todo: check this softmax
        x = F.tanh(x)
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

        # split data into real part and imaginary part
        self.encoderRe = encoder(width//2)
        self.decoderRe = decoder(width//2)

        self.encoderIm = encoder(width//2)
        self.decoderIm = decoder(width//2)

        self.width = width
        self.is_autoencoder = False

        def init_weights(m):
            if type(m) is nn.Linear:
                nn.init.xavier_uniform(m.weight)

        self.apply(init_weights)

    def forward(self, x, y):
        # todo: parse x,y
        if self.is_autoencoder:
            return {"Re": self.decoderRe(self.encoderRe(x)),
                    "Im": self.decoderIm(self.encoderIm(y))}
        else:
            return {"Re": self.decoderRe(x), "Im": self.decoderIm(y)}

    def get_noise_width(self):
        """
        :return: Width of the noise vector that the generator expects when in decoder mode
        """
        return self.width // 16

    def set_autoencoder(self, is_autoencoder):
        self.is_autoencoder = is_autoencoder
        return self