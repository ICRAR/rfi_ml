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
import logging
from torch.utils.data import Dataset

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


class HDF5Dataset(Dataset):
    """
    Creates a dataset to loop over all of the NN inputs contained within an HDF5 file
    that was exported from an LBA file using 'preprocess.py'
    """

    valid_types = {'abs_angle', 'real_imag'}

    @staticmethod
    def get_index(index, length, possibilities):
        part_size = length // len(possibilities)
        return possibilities[index // part_size], index % part_size

    def __init__(self, filename, data_type, **kwargs):
        """
        Initialises the HDF5 dataset
        :param filename: The filename to read the dataset from
        :param data_type: What type of data should be read from the file? 'angle_abs' or 'real_imag'
        :param kwargs:
            polarisations: A list of polarisations (0 to 1) that should be included in the data
            frequencies: A list of frequencies (0 to 3) that should be included in the data
            max_inputs: Maximum number of inputs to get per polarisation, frequency pair
            full_first: Reconstruct the full real / absolute values if set to true
            normalise: Normalise the data as per the dataset min and max values in the HDF5 file
        """
        super(HDF5Dataset, self).__init__()

        self.hdf5 = h5py.File(filename, 'r', swmr=True, libver='latest', rdcc_nbytes=1000000000)  # 1 GB
        self.cache = {}

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
            value = kwargs.get(name, None)
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

        # Size of first part of nn input (real or absolute)
        self.size_first = get_attribute('size_first')

        # Size of second part of nn input (imaginary or angles)
        self.size_second = get_attribute('size_second')

        # User specified they only want these polarisations
        self.polarisations = get_ints_argument('polarisations', [0, 1])

        # User specified they only want these frequencies
        self.frequencies = get_ints_argument('frequencies', [0, 1, 2, 3])

        # User wants the real / abs values to be fully re-created from the half that we have
        self.full_first = kwargs.get('full_first', False)

        # User wants to cache data from the HDF5 file instead of re-reading it
        self.use_cache = kwargs.get('use_cachce', False)

        # User specified they only want to use this many NN inputs from the entire dataset
        self.max_inputs = get_int_argument('max_inputs')
        if self.max_inputs is None:
            self.max_inputs = self.fft_count
        else:
            if self.max_inputs == 0:
                self.max_inputs = self.fft_count
            elif self.max_inputs < 0:
                raise RuntimeError('max_input < 0')
            elif self.max_inputs > self.fft_count:
                LOG.warn('max_inputs > fft_count. Clamping to {0}'.format(self.fft_count))
                self.max_inputs = self.fft_count

        # If true, normalise data before returning it
        self.normalise = kwargs.get('normalise', False)

    def __len__(self):
        """
        :return: Number of inputs in the dataset
        """
        return self.max_inputs * len(self.polarisations) * len(self.frequencies)

    def __getitem__(self, i):
        """
        Gets an NN input from the dataset. This will iterate over all of the selected
        polarisations and frequencies.
        :param index: The index to get.
        :return: A single NN input
        """
        if self.use_cache:
            cached = self.cache.get(i, None)
            if cached is not None:
                return cached

        length = len(self)
        p, index = self.get_index(i, length, self.polarisations)
        c, index = self.get_index(index, length // len(self.polarisations), self.frequencies)
        key = 'p{0}_c{1}_{2}'.format(p, c, self.type)
        data_container = self.hdf5[key]
        data = data_container[index * self.size : (index + 1) * self.size]
        if self.full_first:
            data = self.rebuild_first_part(data)

        if self.normalise:
            data = self.normalise_data(data_container, data)

        if self.use_cache:
            self.cache[i] = data

        return data

    def get_configuration(self):
        """
        Gets the current dataset configuration as a dictionary.
        Useful for printing out the configuration.
        :return: Dataset configuration
        :rtype dict
        """
        return {
            'fft_count': self.fft_count,
            'fft_window': self.fft_window,
            'samples': self.samples,
            'size': self.size,
            'size_first': self.size_first,
            'size_second': self.size_second,
            'polarisations': self.polarisations,
            'frequencies': self.frequencies,
            'full_first': self.full_first,
            'max_inputs': self.max_inputs,
            'normalise': self.normalise,
            'type': self.type,
            'input_size': self.get_input_size()
        }

    def get_polarisation_and_channel(self, index):
        """
        Get the polarisation and channel of the input that would be returned from the
        specified index.
        :param index: The index to get the polarisation and channel from
        :return: The polarisation and channel as a tuple
        """
        length = len(self)
        p, index = self.get_index(index, length, self.polarisations)
        c, index = self.get_index(index, length // len(self.polarisations), self.frequencies)
        return p, c

    def rebuild_first_part(self, data):
        """
        Reconstructs the entire real / absolute values from the half values that are included in the
        HDF5 file
        :param data: Data to reconstruct
        :return: Data with reconstructed first half
        """
        first = data[0:self.size_first]
        return np.concatenate((first, np.flip(first), data[self.size_first:self.size]))

    def normalise_data(self, data_container, data):
        """
        Normalise the provided data with the min and max values in the HDF5 dataset
        :param data_container: The HDF5 dataset that contains min and max values
        :param data: The data to normalise
        :return: Normalised data
        """
        first_size = self.size_first * 2 if self.full_first else self.size_first
        size = first_size + self.size_second
        data1 = data[0:first_size]
        data2 = data[first_size:size]

        # Normalise to -1 to 1
        minimum = data_container.attrs[self.type_minmax_keys[0]]
        maximum = data_container.attrs[self.type_minmax_keys[1]]
        data1 -= minimum
        data1 /= maximum - minimum
        data1 *= 2
        data1 -= 1

        minimum = data_container.attrs[self.type_minmax_keys[2]]
        maximum = data_container.attrs[self.type_minmax_keys[3]]
        data2 -= minimum
        data2 /= maximum - minimum
        data2 *= 2
        data2 -= 1

        return np.concatenate((data1, data2), axis=0)

    def close(self):
        """
        Close the dataset
        """
        self.hdf5.close()

    def get_input_size(self):
        """
        :return: The size of each input that this dataset will return
        """
        return self.get_input_size_first() + self.size_second

    def get_input_size_first(self):
        """
        :return: The size of the first part of the input (real / absolute)
        """
        return self.size_first * 2 if self.full_first else self.size_first

    def precache(self):
        """
        Add all items in the dataset to the cache
        """
        if not self.use_cache:
            raise Exception('Cache is not enabled')

        for i in range(len(self)):
            self.__getitem__(i)


def test():
    sum = 0
    for d in loader:
        sum += d.shape[0]
    return sum


if __name__ == '__main__':
    import timeit
    import time
    from torch.utils.data import DataLoader

    dataset = HDF5Dataset('/home/sam/Projects/rfi_ml/src/gan/At.hdf5', 'abs_angle',
                          use_cachce=True,
                          normalise=True,
                          full_first=True,
                          polarisations=[0, 1],
                          frequencies=[0, 1, 2, 3])

    loader = DataLoader(dataset,
                        batch_size=4096,
                        shuffle=True,
                        pin_memory=False,
                        num_workers=0)

    start = time.time()
    dataset.precache()
    print(time.time() - start)
    print(timeit.timeit('test()', setup='from __main__ import test', number=100))

