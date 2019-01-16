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

import h5py
import numpy as np
from torch.utils.data import Dataset


class LazyHDF5Dataset(Dataset):

    def __init__(self, filename, width):
        """
        Initialise the lazily loaded HDF5 dataset
        :param filename: The filename to load data from
        :param width: Number of values to load for a single net input
        """
        super(LazyHDF5Dataset, self).__init__()

        self.hdf5 = h5py.File(filename, 'r')
        self.data = self.hdf5['data']
        self.width = width

    def __del__(self):
        self.close()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].astype(np.float32)

    def close(self):
        self.hdf5.close()

if __name__ == '__main__':
    dataset = LazyHDF5Dataset('../../data/At_c0_p0_s1000000000_fft2048.hdf5', 2048)
    count = 0
    for i in range(len(dataset) // 100):
        value = dataset[i:i + 100]
        count += 1
    print(count)
    count = 0
    for i in range(len(dataset)):
        value = dataset[i]
        count += 1
    print(count)
