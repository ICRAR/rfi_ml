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
Provides an iterator over and HDF5 dataset and gaussian noise sources for GAN training.
"""

import logging

from torch.utils.data import DataLoader

from .HDF5Dataset import HDF5Dataset
from .NoiseDataset import NoiseDataset

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


class Data(object):

    def __init__(self, filename: str, batch_size: int, **kwargs):
        """
        Provides access to an iterator over a set of training data. The training data consists of data read from an HDF5
        file along with two generated gaussian noise datasets.

        Noise data is provided by `src.NoiseDataset.NoiseDataset` and HDF5 data is provided by
        `src.HDF5Dataset.HDF5Dataset`.

        Data will be in the shape (`src.HDF5Dataset.HDF5Dataset.get_input_shape`, `batch_size`).
        ```python
        data = Data("fft.hdf5", 128)
        for data, noise1, noise2 in data:
            # Do something with data loaded from "fft.hdf5"
            # Assuming each input in fft.hdf5 is of size 1024, then the shape of
            # data, noise1, and noise2 is (1024, 128)
            module(data)
        ```

        The data set can be iterated multiple times and the iteration order will be random each time.

        Parameters
        ----------
        filename : str
            The HDF5 file to load the data from.
        batch_size : int
            Number of NN inputs to include in each batch.
        kwargs
            Keyword arguments to pass to the `HDF5Dataset` constructor.
        """
        self.hdf5_dataset = HDF5Dataset(filename, **kwargs)

        LOG.info('Dataset params: {0}'.format(self.hdf5_dataset.get_configuration()))
        LOG.info('Data loader params: {0}'.format({
            'filename': filename,
            'batch_size': batch_size
        }))
        LOG.info('Number of NN inputs available with this configuration {0}'.format(len(self.hdf5_dataset)))

        width = self.get_input_shape()[0]

        self.data = DataLoader(self.hdf5_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               pin_memory=False,
                               num_workers=0)

        self.noise1 = DataLoader(NoiseDataset(width, len(self.hdf5_dataset)),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=False,
                                 num_workers=0)

        self.noise2 = DataLoader(NoiseDataset(width, len(self.hdf5_dataset)),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=False,
                                 num_workers=0)

    def __len__(self):
        """
        Get the number of batches in the dataset.
        :return: Number of batches in the dataset.
        :rtype int
        """
        return len(self.data)

    def __iter__(self):
        """
        Get an iterator over the data in the dataset.
        :return: Iterator over the data, noise1, and noise2 datasets.
        :rtype iter
        """
        return zip(self.data, self.noise1, self.noise2)

    def get_input_shape(self):
        """
        Get the shape of a single input returned from this dataset

        Returns
        -------
        int: Width of a single input
        """
        return self.hdf5_dataset.get_input_shape()

    def close(self):
        """
        Close the underlying HDF5 dataset
        """
        self.hdf5_dataset.close()
