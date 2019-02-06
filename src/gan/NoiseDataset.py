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

import numpy as np
from torch.utils.data import Dataset


def generate_fake_noise(inputs, size):
    """
    Generate fake noise
    Generate gaussian noise using -0.0289923828125, 1.9391296947313124 as mean and stddev. These are the
    mean and stdddev of the lba files.
    TODO: Actually generate -3, -1, 1, 3 as the only pieces of data
    TODO: Unused.
    :param inputs:
    :param size:
    :return:
    """
    return np.random.normal(-0.0289923828125, 1.9391296947313124, (inputs, size)).astype(np.float32)


class NoiseDataset(Dataset):
    """
    Generates gaussian noise on the fly.
    """

    def __init__(self, width, length):
        """
        Create a new noise dataset
        :param int width: With of the noise.
        :param int length: Size of the dataset. This is needed for Pytorch data loaders to operate correctly.
        """
        self.width = width
        self.length = length

    def __getitem__(self, item):
        """
        Generate a new input of gaussian noise.
        :param int item: Needed for __getitem__. Unused.
        :return np.array: Numpy array containing the data
        """
        data = np.random.normal(0, 1.0, self.width).astype(np.float32)
        return data

    def __len__(self):
        """
        Get the length of the dataset.
        THis is needed for Pytorch data loaders to operate correctly.
        :return: Length of the noise dataset
        :rtype int
        """
        return self.length
