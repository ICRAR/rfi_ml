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
import matplotlib.pyplot as plt
import itertools
import numpy as np
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):

    valid_types = {'abs_angle', 'real_imag'}

    @staticmethod
    def get_index(index, length, possibilities):
        part_size = length // len(possibilities)
        return possibilities[index // part_size], index % part_size

    def __init__(self, filename, data_type, **kwargs):
        """
        """
        super(HDF5Dataset, self).__init__()

        self.hdf5 = h5py.File(filename, 'r')

        def get_attribute(name):
            value = self.hdf5.attrs.get(name, None)
            if value is None:
                raise RuntimeError('Invalid HDF5 file. {0} was not set in attributes'.format(name))
            return value

        def get_ints_argument(name, default):
            value = kwargs.get(name, None)
            if value is None:
                return default
            else:
                t = type(value)
                if t != list and t != int:
                    raise RuntimeError('{0} must be a list of integers, or an integer'.format(name))
                if t == list and not all(type(p) == int for p in value):
                    raise RuntimeError('{0} list must contain only integers'.format(name))
            return [value] if type(value) is int else value

        def get_int_argument(name):
            value = kwargs.get('max_inputs', None)
            if value is not None and type(value) != int:
                raise RuntimeError('{0} should be an integer'.format(name))
            return value

        # Data type to extract from the hdf5
        self.type = data_type
        if self.type not in self.valid_types:
            raise RuntimeError('type should be one of {0}, not {1}'.format(self.valid_types, self.type))

        # Keys for getting the attribute names of the min and max values
        self.type_minmax_keys = ['{0}_{1}'.format(m, k) for k, m in itertools.product(self.type.split('_'), ['min', 'max'])]

        # Number of FFTs contained within the file
        self.fft_count = get_attribute('fft_count')

        # Width of the FFTs contained within the file
        self.fft_window = get_attribute('fft_window')

        # Total number of samples that were FFT'd
        self.samples = get_attribute('samples')

        # Size of each NN input contained within this file.
        self.size = get_attribute('size')

        # User specified they only want these polarisations
        self.polarisations = get_ints_argument('polarisations', [0, 1])

        # User specified they only want these frequencies
        self.frequencies = get_ints_argument('frequencies', [0, 1, 2, 3])

        # User specified they only want to use this many NN inputs from the entire dataset
        self.max_inputs = get_int_argument('max_inputs')
        if self.max_inputs is None:
            self.max_inputs = self.fft_count
        else:
            if self.max_inputs <= 0:
                raise RuntimeError('max_input <= 0')
            # Don't go over the fft_count limit
            self.max_inputs = min(self.max_inputs, self.fft_count)

        # If true, normalise data before returning it
        self.normalise = kwargs.get('normalise', False)

    def __del__(self):
        self.close()

    def __len__(self):
        return self.max_inputs * len(self.polarisations) * len(self.frequencies)

    def __getitem__(self, index):
        length = len(self)
        p, index = self.get_index(index, length, self.polarisations)
        c, index = self.get_index(index, length // len(self.polarisations), self.frequencies)
        key = 'p{0}_c{1}_{2}'.format(p, c, self.type)
        data_container = self.hdf5[key]
        data = data_container[index * self.size : (index + 1) * self.size]
        if self.normalise:
            data = self.normalise_data(data_container, data)

        return p, c, data

    def normalise_data(self, data_container, data):

        half_size = self.size // 2
        data1 = data[0:half_size]
        data2 = data[half_size:self.size]

        # Normalise to -1 to 1
        data1 -= data_container.attrs[self.type_minmax_keys[0]]
        data1 /= data_container.attrs[self.type_minmax_keys[1]]
        data1 *= 2
        data1 -= 1

        data2 -= data_container.attrs[self.type_minmax_keys[2]]
        data2 /= data_container.attrs[self.type_minmax_keys[3]]
        data2 *= 2
        data2 -= 1

        return np.concatenate((data1, data2), axis=0)

    def close(self):
        self.hdf5.close()


if __name__ == '__main__':
    dataset = HDF5Dataset('/home/sam/at_data_2048.hdf5', 'real_imag',
                          normalise=True,
                          max_inputs=1,
                          polarisations=[0],
                          frequencies=[0, 2, 3])
    count_p = {}
    count_c = {}
    for i in range(len(dataset)):
        p_value, c_value, data = dataset[i]
        count_p[p_value] = count_p.get(p_value, 0) + 1
        count_c[c_value] = count_c.get(c_value, 0) + 1
        print(p_value, c_value, data.shape)
        fig = plt.figure()
        plt.plot(data)
        plt.show()
        plt.close(fig)
    print(count_p)
    print(count_c)
    print(len(dataset))
