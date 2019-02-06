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

import sys
import os
base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

import logging
from torch.utils.data import DataLoader
from gan.HDF5Dataset import HDF5Dataset
from gan.NoiseDataset import NoiseDataset

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


class Data(object):
    """
    Provides access to an iterator over a set of training data. The training data consists of data read from an HDF5 file.
    along with two generated gaussian noise datasets
    """

    def __init__(self, filename, data_type, batch_size, **kwargs):
        """
        Create a new dataset from the provided HDF5 file.
        :param str filename: The HDF5 file to load the data from.
        :param str data_type: Type of data to return from the HDF5 file.  See :func:`~gan.HDF5Dataset`
        :param int batch_size: Number of NN inputs to include in each batch.
        :param kwargs: kwargs to pass to the :func:`~gan.HDF5Dataset` constructor.
        """
        self.hdf5_dataset = HDF5Dataset(filename, data_type, **kwargs)

        LOG.info('Dataset params: {0}'.format(self.hdf5_dataset.get_configuration()))
        LOG.info('Data loader params: {0}'.format({
            'filename': filename,
            'batch_size': batch_size,
        }))
        LOG.info('Number of NN inputs available with this configuration {0}'.format(len(self.hdf5_dataset)))

        self.data = DataLoader(self.hdf5_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               pin_memory=False,
                               num_workers=0)

        self.noise1 = DataLoader(NoiseDataset(self.get_input_size(), len(self.hdf5_dataset)),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=False,
                                 num_workers=0)

        self.noise2 = DataLoader(NoiseDataset(self.get_input_size(), len(self.hdf5_dataset)),
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

    def get_input_size(self):
        """
        Get the width of a single NN input that will be returned by this dataset.
        :return: Width of a single input
        :rtype int
        """
        return self.hdf5_dataset.get_input_size()

    def get_input_size_first(self):
        """
        Gets the width of the first part of the input (real / absolute values)
        This may be half of the size of the second input part, because the real / absolute values are
        mirrored around their centre.
        :return: Width of the first part of the input
        :rtype int
        """
        return self.hdf5_dataset.get_input_size_first()

    def close(self):
        """
        Close the underlying HDF5 dataset
        """
        self.hdf5_dataset.close()
