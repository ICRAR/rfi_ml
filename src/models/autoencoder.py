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

import torch
import torch.nn as nn
from enum import Enum


class Autoencoder(nn.Sequential):
    """
    Autoencoder to perform dimension reduction on input data before it's fed into the discriminator or generator.
    """

    class Mode(Enum):
        """
        Defines the possible modes the autoencoder can be in
        """
        AUTOENCODER = 0
        ENCODER = 1
        DECODER = 2

    def __init__(self, width):
        """
        Construct the autoencoder
        :param int width: Width of the autoencoder input
        """
        super(Autoencoder, self).__init__()

        def layer(in_size, out_size, dropout, activation, final=False):
            layers = [
                nn.Linear(in_size, out_size),
                #activation()
            ]

            if not final:
                pass
                #layers.append(nn.AlphaDropout(dropout))

            return layers

        hidden1 = int(width * 0.95)
        hidden2 = int(width * 0.9)
        hidden3 = int(width * 0.85)
        hidden4 = int(width * 0.8)

        self.encoder = nn.Sequential(
            *layer(width, width, 0.0, nn.SELU),
            #*layer(hidden1, hidden2, 0.1, nn.SELU),
            #*layer(hidden2, hidden3, 0.1, nn.SELU),
            #*layer(hidden3, hidden4, 0.1, nn.SELU),
        )

        self.decoder = nn.Sequential(
            #*layer(hidden4, hidden3, 0.1, nn.SELU),
            #*layer(hidden3, hidden2, 0.1, nn.SELU),
            #*layer(hidden2, hidden1, 0.1, nn.SELU),
            *layer(width, width, 0.0, nn.SELU, final=True),
        )

        """
        self.encoder_absolute = nn.Sequential(
            *layer(width, hidden1, 0.0, nn.ReLU),
            *layer(hidden1, hidden2, 0.0, nn.ReLU),
            *layer(hidden2, hidden3, 0.0, nn.ReLU),
            *layer(hidden3, hidden4, 0.0, nn.ReLU),
        )

        self.encoder_angle = nn.Sequential(
            *layer(width, hidden1, 0.0, nn.Softsign),
            *layer(hidden1, hidden2, 0.0, nn.Softsign),
            *layer(hidden2, hidden3, 0.0, nn.Softsign),
            *layer(hidden3, hidden4, 0.0, nn.Softsign),
        )

        self.decoder_absolute = nn.Sequential(
            *layer(hidden4, hidden3, 0.0, nn.ReLU),
            *layer(hidden3, hidden2, 0.0, nn.ReLU),
            *layer(hidden2, hidden1, 0.0, nn.ReLU),
            *layer(hidden1, width, 0.0, nn.ReLU),
        )

        self.decoder_angle = nn.Sequential(
            *layer(hidden4, hidden3, 0.0, nn.Softsign),
            *layer(hidden3, hidden2, 0.0, nn.Softsign),
            *layer(hidden2, hidden1, 0.0, nn.Softsign),
            *layer(hidden1, width, 0.0, nn.Softsign),
        )

        self.encoder_dense = nn.Sequential(
            *layer(hidden2 * 2, hidden3, 0.1)
        )

        self.decoder_dense = nn.Sequential(
            *layer(hidden3, hidden2 * 2, 0.1)
        )
        """

        self.encoder_width = width
        self.decoder_width = hidden2
        self.mode = self.Mode.AUTOENCODER

        def init_weights(m):
            if type(m) is nn.Linear:
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def forward(self, x):
        """
        Calculate the autoencoder's output for x, given its current mode
        :param Tensor x: Tensor to calculate output for. Must be the appropriate width: get_encoder_input_width()
                         if in AUTOENCODER or ENCODER mode, and get_decoder_input_width() if in DECODER mode.
        """
        if self.mode == self.Mode.AUTOENCODER:

            """
            encoded_absolute = self.encoder_absolute(x[:, 0])
            encoded_angle = self.encoder_angle(x[:, 1])

            
            dense = torch.cat((encoded_absolute, encoded_angle), dim=1)
            encoded_dense = self.encoder_dense(dense)
            decoded_dense = self.decoder_dense(encoded_dense)

            size = decoded_dense.size()[1]
            absolute = decoded_dense[:, 0:size // 2]
            angle = decoded_dense[:, size // 2:size]
            
            decoded_absolute = self.decoder_absolute(encoded_absolute)
            decoded_angle = self.decoder_angle(encoded_angle)

            out = torch.stack((decoded_absolute, decoded_angle), dim=1)
            """
            out = self.decoder(self.encoder(x))

            return out

        elif self.mode == self.Mode.ENCODER:
            return self.encoder(x)
        elif self.mode == self.Mode.DECODER:
            return self.decoder(x)
        else:
            raise RuntimeError("Bad autoencoder mode")

    def set_mode(self, mode):
        """
        Sets the mode of the autoencoder.
        :param self.Mode mode: The mode to use.
        """
        if mode not in self.Mode:
            raise RuntimeError("Bad autoencoder mode")
        self.mode = mode

    def get_mode(self):
        """
        :return: The current mode of the autoencoder.
        :rtype self.Mode
        """
        return self.mode

    def get_encoder_input_width(self):
        """
        :return: Required input width in AUTOENCODER or ENCODER mode.
        :rtype int
        """
        return self.encoder_width

    def get_decoder_input_width(self):
        """
        :return: Required input width in DECODER mode.
        :rtype int
        """
        return self.decoder_width



