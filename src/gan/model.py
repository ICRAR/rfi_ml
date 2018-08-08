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
Seed the generator with gaussian noise of varying sizes. This will end up being whats used to generate new rfi.
After the convolution step in the discriminator, pass in the fft values as two arrays (one real, one imaginary) concatted together with the results of the convolution step.

Discriminator: Determines if the provided input is actually RFI. (signal -> 1, 0 or 0, 1)
Generator: Generates RFI by taking gaussian noise and producing RFI. (gaussian noise -> signal)
"""
from torch import nn
from collections import namedtuple

InputDetails = namedtuple('InputDetails', 'size_in size_out')


class Discriminator(nn.Module):
    """
    Determines whether the provided input is actually RFI noise
    """

    INPUT_DETAILS = {
        256: InputDetails(238, 0),
        512: InputDetails(494, 0),
        1024: InputDetails(1006, 0)
    }

    def __init__(self, sample_size):
        super(Discriminator, self).__init__()
        # Convolution part to analyse the signal itself (signal normalised from -3 to 3 into -1 to 1
        self.convolution = nn.Sequential(
            nn.Conv1d(1, 10, 7),
            nn.ELU(),
            nn.BatchNorm1d(10),
            nn.Conv1d(10, 10, 5),
            nn.ELU(),
            nn.BatchNorm1d(10),
            nn.Conv1d(10, 10, 5),
            nn.ELU(),
            nn.BatchNorm1d(10),
            nn.Conv1d(10, 10, 5),
            nn.ELU(),
        )
        in_size = 10 * self.INPUT_DETAILS[sample_size].size_in
        self.linear = nn.Sequential(
            nn.Linear(in_size, sample_size),
            nn.ELU(alpha=0.3),
            nn.Dropout(p=0.4),
            nn.Linear(sample_size, sample_size // 4),
            nn.ELU(alpha=0.3),
            nn.Linear(sample_size // 4, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = x.view(x.size(0), 1, -1)
        x = self.convolution(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class DiscriminatorFFT(nn.Module):
    """
    Determines if the provided FFT of a signal is from RFI
    """

    INPUT_DETAILS = {
        1024: InputDetails(2048, 0)
    }

    def __init__(self, sample_size):
        super(DiscriminatorFFT, self).__init__()
        in_size = self.INPUT_DETAILS[sample_size].size_in
        self.linear = nn.Sequential(
            nn.Linear(in_size, in_size // 2),
            nn.BatchNorm1d(in_size // 2),
            nn.ELU(alpha=0.3),
            nn.Dropout(p=0.4),

            nn.Linear(in_size // 2, in_size // 4),
            nn.BatchNorm1d(in_size // 4),
            nn.ELU(alpha=0.3),
            nn.Dropout(p=0.4),

            nn.Linear(in_size // 4, in_size // 8),
            nn.BatchNorm1d(in_size // 8),
            nn.ELU(alpha=0.3),
            nn.Dropout(p=0.4),

            nn.Linear(in_size // 8, in_size // 16),
            nn.BatchNorm1d(in_size // 16),
            nn.ELU(alpha=0.3),
            nn.Dropout(p=0.4),

            nn.Linear(in_size // 16, in_size // 32),
            nn.BatchNorm1d(in_size // 32),
            nn.ELU(alpha=0.3),
            nn.Dropout(p=0.4),

            nn.Linear(in_size // 32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.linear(x)


class Generator(nn.Module):
    """
    Generator autoencoder that will receive an array of gaussian noise, and will convert it into RFI noise.
    """

    INPUT_DETAILS = {
        256: InputDetails(232, 0),
        512: InputDetails(488, 0),
        1024: InputDetails(24, 1000)
    }

    def __init__(self, sample_size):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 5, 5),
            nn.ELU(),
            nn.BatchNorm1d(5),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(5, 10, 5),
            nn.ELU(),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(10, 10, 5),
            nn.ELU(),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(10, 10, 5),
            nn.ELU(),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(10, 10, 5),
            nn.ELU(),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(10, 10, 5),
            nn.ELU(),
            nn.Dropout(p=0.4),
        )

        size_in = 10 * self.INPUT_DETAILS[sample_size].size_in
        size_hidden = 10 * (sample_size // 4)
        size_out = 10 * self.INPUT_DETAILS[sample_size].size_out

        self.linear = nn.Sequential(
            nn.Linear(size_in, size_hidden),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(size_hidden),

            nn.Linear(size_hidden, size_hidden),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(size_hidden),

            nn.Linear(size_hidden, size_out),
            nn.ELU(),
            nn.Dropout(p=0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(10, 10, 5),
            nn.ELU(),
            nn.BatchNorm1d(10),

            nn.ConvTranspose1d(10, 10, 5),
            nn.ELU(),
            nn.BatchNorm1d(10),

            nn.ConvTranspose1d(10, 10, 5),
            nn.ELU(),
            nn.BatchNorm1d(10),

            nn.ConvTranspose1d(10, 10, 5),
            nn.ELU(),
            nn.BatchNorm1d(10),

            nn.ConvTranspose1d(10, 5, 5),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(5),

            nn.ConvTranspose1d(5, 1, 5),
            nn.ELU(),
            nn.BatchNorm1d(1),
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.view(x.size(0), 10, -1)
        x = self.decoder(x)
        x = x.view(x.size(0), -1)
        return x


def get_models(sample_size):
    discriminator = DiscriminatorFFT(sample_size)
    generator = Generator(sample_size)

    return discriminator, generator
