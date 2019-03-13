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
    type_index_map = {'real': 0, 'imag': 1, 'abs': 2, 'angle': 3}

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

        types = self.type.split('_')
        self.first_type_index = self.type_index_map[types[0]]
        self.second_type_index = self.type_index_map[types[1]]

        # Keys for getting the attribute names of the min and max values
        self.type_minmax_keys = ['{0}_{1}'.format(m, k) for k, m in itertools.product(types, ['min', 'max'])]

        # Number of FFTs contained within the file
        self.fft_count = get_attribute('fft_count')

        # Width of the FFTs contained within the file
        self.fft_window = get_attribute('fft_window')

        # Total number of samples that were FFT'd
        self.samples = get_attribute('samples')

        # Size of each NN input
        self.input_size = get_attribute('input_size')

        # User specified they only want these polarisations
        self.polarisations = get_ints_argument('polarisations', [0, 1])

        # User specified they only want these frequencies
        self.frequencies = get_ints_argument('frequencies', [0, 1, 2, 3])

        # User wants to cache data from the HDF5 file instead of re-reading it
        self.use_cache = kwargs.get('use_cache', False)

        # Slice data_slice[0] elements from the start of the input, and data_slice[1] from the end.
        # This can be used to remove FFT artifacts
        self.data_slice = kwargs.get('data_slice', (0, 0))
        slice_type = type(self.data_slice)
        if slice_type == tuple:
            length = len(self.data_slice)
            if length == 1:
                self.data_slice = (self.data_slice[0], 0)
            elif length != 2:
                raise RuntimeError('data_slice must be a tuple of length 1 or 2')
            if type(self.data_slice[0]) != int or type(self.data_slice[1]) != int:
                raise RuntimeError('data_slice elements must be ints')
        elif slice_type == int:
            self.data_slice = (self.data_slice, self.data_slice)
        else:
            raise RuntimeError('data_slice must be a tuple or an int')

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
                LOG.warning('max_inputs > fft_count. Clamping to {0}'.format(self.fft_count))
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
        key = 'p{0}_c{1}'.format(p, c)
        data_container = self.hdf5[key]
        data_first = data_container[index][self.first_type_index]
        data_first = data_first[self.data_slice[0]:data_first.shape[0] - self.data_slice[1]]
        data_second = data_container[index][self.second_type_index]
        data_second = data_second[self.data_slice[0]:data_second.shape[0] - self.data_slice[1]]

        if self.normalise:
            # Normalise to -1 to 1
            minimum = data_container.attrs[self.type_minmax_keys[0]]
            maximum = data_container.attrs[self.type_minmax_keys[1]]
            data_first -= minimum
            data_first /= maximum - minimum
            data_first *= 2
            data_first -= 1

            minimum = data_container.attrs[self.type_minmax_keys[2]]
            maximum = data_container.attrs[self.type_minmax_keys[3]]
            data_second -= minimum
            data_second /= maximum - minimum
            data_second *= 2
            data_second -= 1

        data = np.concatenate((data_first, data_second))

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
            'polarisations': self.polarisations,
            'frequencies': self.frequencies,
            'max_inputs': self.max_inputs,
            'normalise': self.normalise,
            'type': self.type,
            'input_size': self.get_input_size(),
            'first_type_index': self.first_type_index,
            'second_type_index': self.second_type_index
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

    def close(self):
        """
        Close the dataset
        """
        self.hdf5.close()

    def get_input_size(self):
        """
        :return: The size of each input that this dataset will return
        """
        size = int(self.input_size * 2) - (self.data_slice[0] + self.data_slice[1])
        return size

    def get_labels(self):
        """
        Get the data type labels for the data inputs.
        e.g. if gan_config.settings.DATA_TYPE == 'real_imag', this will return ['real', 'imag']
        :return: List of two labels for the data inputs.
        :rtype list
        """
        return self.type.split('_')

    def precache(self):
        """
        Add all items in the dataset to the cache
        """
        if not self.use_cache:
            LOG.warning('precache: Cache is not enabled')
            return

        for i in range(len(self)):
            self.__getitem__(i)


if __name__ == '__main__':
    import timeit
    import time
    from torch.utils.data import DataLoader

    current_loader = None

    def test():
        sum = 0
        global current_loader
        for d in current_loader:
            sum += d.shape[0]
            return sum


    def test_file(name):
        dataset = HDF5Dataset(name, 'abs_angle',
                              # use_cache=True,
                              normalise=True,
                              full_first=True,
                              polarisations=[0, 1],
                              frequencies=[0, 1, 2, 3])

        global current_loader
        current_loader = DataLoader(dataset,
                                    batch_size=4096,
                                    shuffle=True,
                                    pin_memory=False,
                                    num_workers=0)

        LOG.info("Test for file: {0}".format(name))
        LOG.info("Dataset params: {0}".format(dataset.get_configuration()))

        start = time.time()
        LOG.info("Start time: {0}".format(start))
        dataset.precache()
        LOG.info("Precache time: {0}".format(time.time() - start))
        t = timeit.timeit('test()', setup='from __main__ import test', number=30)
        LOG.info("Average iteration time: {0}".format(t / 30))

    for f in ['At.hdf5', 'At_1_1_size.hdf5', 'At_auto.hdf5', 'At_none.hdf5']:
        test_file(f)
