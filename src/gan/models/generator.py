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

import torch.nn as nn


class Generator(nn.Sequential):
    """
    Generator to generate RFI from a noise vector
    """

    def __init__(self, in_width, out_width):
        """
        Construct the generator
        """

        def layer(in_size, out_size, final=False):
            layers = [nn.Linear(in_size, out_size)]

            if not final:
                layers.append(nn.Softsign())
                layers.append(nn.Dropout(0.1))

            return layers

        def lerp(x, y, t):
            return x * (1 - t) + y * t

        hidden1 = int(lerp(in_width, out_width, 0.25))
        hidden2 = int(lerp(in_width, out_width, 0.50))
        hidden3 = int(lerp(in_width, out_width, 0.75))

        super(Generator, self).__init__(
            *layer(in_width, hidden1),
            *layer(hidden1, hidden2),
            *layer(hidden2, hidden3),
            *layer(hidden2, hidden3),
            *layer(hidden3, out_width),
        )

        self.in_width = in_width
        self.out_width = out_width

        def init_weights(m):
            if type(m) is nn.Linear:
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def get_input_width(self):
        """
        :return: Width of the noise vector that the generator expects when in decoder mode
        :rtype int
        """
        return self.in_width

    def get_output_width(self):
        """
        :return: Width of the output of the generator
        :rtype int
        """
        return self.out_width
