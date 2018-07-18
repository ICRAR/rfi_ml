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
import torch
from torch import nn


class Discriminator(nn.Module):
    """
    Determines whether the provided input is actually RFI noise
    """
    def __init__(self, sample_size, fft_size):
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
        # Linear layer to make the decision.
        # This layer will accept both the result of the signal convolution, and an FFT of the signal
        # (containing an array of real values followed by the imaginary values)
        # fft_size is the number of real + imaginary values
        in_size = sample_size - 18 + fft_size
        out_size = sample_size + fft_size
        self.linear = nn.Sequential(
            nn.Linear(in_size, out_size),
            # Considering this is also taking the fft, it may need to be made wider / deeper
            nn.ELU(alpha=0.3),
            nn.Dropout(p=0.4),
            nn.Linear(out_size, out_size // 4),
            nn.ELU(alpha=0.3),
            nn.Linear(out_size // 4, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x, fft):
        """
        :param x:
        :param fft: fftr + ffti flattened.
        :return:
        """
        x = x.view(x.size(0), 1, -1)
        x = self.convolution(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, fft), dim=1)
        x = self.linear(x)
        return x


class Generator(nn.Module):
    """
    Generator autoencoder that will receive an array of gaussian noise, and will convert it into RFI noise.
    """

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

        # Linear maps from in to hidden to out (which is same size as in)
        size_in = sample_size // 32
        size_hidden = sample_size // 4
        size_out = size_in
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
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.view(x.size(0), 10, -1)
        x = self.decoder(x)
        x = x.view(x.size(0), -1)
        return x
