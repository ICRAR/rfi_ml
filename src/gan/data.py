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

import torch
import logging
from torch.utils.data import DataLoader
from gan.HDF5Dataset import HDF5Dataset
from gan.NoiseDataset import NoiseDataset

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


class Data(object):
    def __init__(self, filename, data_type, batch_size, **kwargs):
        self.hdf5_dataset = HDF5Dataset(filename, data_type, **kwargs)

        LOG.info('Dataset params: {0}'.format(self.hdf5_dataset.get_configuration()))
        LOG.info('Data loader params: {0}'.format({
            'filename': filename,
            'batch_size': batch_size,
        }))

        self.data = DataLoader(self.hdf5_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               pin_memory=False,
                               num_workers=0)

        self.noise1 = DataLoader(NoiseDataset(self.get_input_size()),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=False,
                                 num_workers=0)

        self.noise2 = DataLoader(NoiseDataset(self.get_input_size()),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=False,
                                 num_workers=0)

    def __iter__(self):
        return zip(self.data, self.noise1, self.noise2)

    def generate_labels(self, num_samples, pattern, use_cuda):
        var = torch.FloatTensor([pattern] * num_samples)
        return var.cuda() if use_cuda else var

    def get_input_size(self):
        return self.hdf5_dataset.get_input_size()

    def get_input_size_first(self):
        return self.hdf5_dataset.get_input_size_first()

    def close(self):
        self.hdf5_dataset.close()
