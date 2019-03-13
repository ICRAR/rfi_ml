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


class Discriminator(nn.Sequential):
    """
    Determines whether the provided input is actually RFI noise
    """

    def __init__(self, width):
        """
        Construct the discriminator
        """
        def layer(in_size, out_size, final=False):
            layers = [nn.Linear(in_size, out_size)]

            if final:
                # layers.append(nn.Softmax(dim=1))
                layers.append(nn.Sigmoid())  # 0 - 1 for single labels
            else:
                layers.append(nn.Softsign())
                layers.append(nn.Dropout(0.1))

            return layers

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

        self.width = width

        def init_weights(m):
            if type(m) is nn.Linear:
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def get_width(self):
        """
        :return: The required width of the input.
        :rtype int
        """
        return self.width
