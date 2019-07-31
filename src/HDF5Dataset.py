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
Load a pytorch dataset from an HDF5 file stored in the HDF5 FFT format.
"""

import logging
import math
import numpy as np
from torch.utils.data import Dataset

from .preprocess.fft.hdf5_fft_definition import HDF5FFTDataSet

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


# TODO: Allow specification of specific channels to use
class HDF5Dataset(Dataset):

    def __init__(self, filename: str, **kwargs):
        """
        Creates a dataset to loop over each network input stored in an HDF5 FFT file.

        Each input is a 2D array of values containing the absolute and angle FFT values of size (fft_size, 2).
        Where the X dimension contains the FFT data, Y[0] contains the absolute values and Y[1] contains the angle
        values.

        Parameters
        ----------
        filename : string
            The filename to read the dataset from
        **kwargs
            `horizontal_concatenate`: Output is concatenated into a single array of size (fft_size * 2, ) in which the
            absolute values and angle values are concatenated together.

            `use_cache`: True to cache data read from the HDF5. False to not

            `max_inputs`: Maximum number of inputs to read from the HDF5 file.

            `normalise`: Normalise the data as per the dataset min and max values in the HDF5 file.
        """
        super(HDF5Dataset, self).__init__()

        self.hdf5 = HDF5FFTDataSet(filename, mode='r', swmr=True, libver='latest', rdcc_nbytes=1000000000)  # 1 GB
        self.cache = {}

        # Number of FFTs contained within the file
        self.fft_count = self.hdf5.fft_count

        # Width of the FFTs contained within the file
        self.fft_window = self.hdf5.fft_window

        # Size of each NN input
        self.input_size = self.hdf5.fft_input_size

        # User wants to return the abs and angle values for each FFT concatenated horzontally to form one
        # big input tensor of size self.input_size * 2
        self.horizontal_concatenate = kwargs.get('horizontal_concatenate', False)
        if self.horizontal_concatenate:
            self.input_size *= 2  # double input size as we're returning abs and angle concat together

        self.num_channels = self.hdf5.num_channels

        # User wants to cache data from the HDF5 file instead of re-reading it
        self.use_cache = kwargs.get('use_cache', False)

        # User specified they only want to use this many NN inputs from the entire dataset
        self.max_inputs = kwargs.get('max_inputs', None)
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
        return self.max_inputs * self.num_channels

    def __getitem__(self, i):
        if self.use_cache:
            cached = self.cache.get(i, None)
            if cached is not None:
                return cached

        # Out of the entire dataset, work out the channel that this index corresponds to.
        channel_index = math.floor(i / self.max_inputs)
        data_index = i % self.max_inputs
        channel = self.hdf5.get_channel(channel_index)

        # This returns an array with an X dimension of 1, as we're only asking for 1 input
        data = channel.read_data(data_index, 1)[0]

        # [:, 0] is abs, [:, 1] is angles

        if self.normalise:
            # Normalise to -1 to 1
            minimum = channel.min_abs
            maximum = channel.max_abs
            data[:, 0] -= minimum
            data[:, 0] /= maximum - minimum
            data[:, 0] *= 2
            data[:, 0] -= 1

            # Normalise to -1 to 1
            minimum = channel.min_angle
            maximum = channel.max_angle
            data[:, 1] -= minimum
            data[:, 1] /= maximum - minimum
            data[:, 1] *= 2
            data[:, 1] -= 1

        if self.horizontal_concatenate:
            data = np.concatenate(data, axis=0)

        if self.use_cache:
            self.cache[i] = data

        return data

    def get_configuration(self) -> dict:
        """
        Gets the current dataset configuration as a dictionary.
        Useful for printing out the configuration.

        Returns
        ----------
        dict: Dataset configuration
        """
        return {
            'fft_count': self.fft_count,
            'fft_window': self.fft_window,
            'max_inputs': self.max_inputs,
            'normalise': self.normalise,
            'horizontal_concatenate': self.horizontal_concatenate,
            'input_shape': self.get_input_shape(),
        }

    def close(self):
        """
        Close the dataset
        """
        self.hdf5.close()

    def get_input_shape(self) -> tuple:
        """
        Get the shape of each input returned by this dataset

        Returns
        ----------
        int: Input shape tuple.
        """
        if self.horizontal_concatenate:
            return int(self.input_size),
        else:
            return int(self.input_size), 2  # 0 is abs, 1 is angle

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
        dataset = HDF5Dataset(name,
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

    for f in ['At_1000.hdf5']:
        test_file(f)
