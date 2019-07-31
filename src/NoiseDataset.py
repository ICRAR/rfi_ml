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
A pytorch dataset that generates gaussian noise in a desired shape.
"""

import numpy as np
from torch.utils.data import Dataset


def _generate_fake_noise(inputs, size):
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

    def __init__(self, width: int, length: int):
        """
        Generates gaussian noise on the fly.

        Implements `__getitem__` and `__len__`
        ```python
        d = NoiseDataset(10, 1000)

        # 1000
        len(d)

        # ndarray(10,)
        d[0]
        ```

        Parameters
        ----------
        width : int
            Width of the noise.
        length : int
            Size of the dataset. This is needed for Pytorch data loaders to operate correctly.
        """
        self._width = width
        self._length = length

    def __getitem__(self, item: int) -> np.ndarray:
        """
        Generate a new input of gaussian noise.

        Parameters
        ----------
        item : int
            Unused

        Returns
        -------
        Numpy array containing the data, with the shape (width,)
        """
        data = np.random.normal(0, 1.0, self._width).astype(np.float32)
        return data

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        This is needed for Pytorch data loaders to operate correctly.

        Returns
        -------
        int: Length of the noise dataset
        """
        return self._length
