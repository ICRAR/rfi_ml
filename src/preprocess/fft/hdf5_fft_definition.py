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
Defines the spec for an `src.preprocess.fft.hdf5_fft_definition.HDF5FFTDataSet` file.
"""

import h5py
import numpy as np
from ..hdf5_utils import Attribute, get_attr, set_attr

MIN_ANGLE = Attribute('min_angle', float)
MAX_ANGLE = Attribute('max_angle', float)
MIN_ABS = Attribute('min_abs', float)
MAX_ABS = Attribute('max_abs', float)

FFT_WINDOW = Attribute('fft_window', int)
FFT_COUNT = Attribute('fft_count', int)
FFT_INPUT_SIZE = Attribute('input_size', int)


class HDF5FFTChannel(object):
    def __init__(self, name: str, dataset: h5py.Dataset):
        """
        Manages a common interface for writing channels to HDF5 FFT datasets to meet the spec.

        Parameters
        ----------
        name : str
            The name of the channel
        dataset : h5py.Dataset
            The dataset that corresponds to this channel
        """
        self._name = name
        self._dataset = dataset

    def write_data(self, offset: int, data: np.ndarray):
        """
        Write FFT data to the channel

        Parameters
        ----------
        offset : int
            Offset in the channel to put the data
        data : np.ndarray
            2D numpy array of FFT values. x = fft index, y = fft values
            e.g. (2, 128) would be 2 ffts of 128 length
        """

        self._dataset[offset:offset + data.shape[0]] = data

    def read_data(self, offset: int, count: int):
        """
        Read one or more FFTs from the channel

        Parameters
        ----------
        offset : int
            Offset to start reading
        count : int
            Number of FFTs to read

        Returns
        -------
        np.ndarray: Numpy array containing as many ffts as are available up to the desired count.
        """
        return self._dataset[offset:offset + count]

    @property
    def min_angle(self):
        """
        The minimum angle value from all FFTs in this channel.
        """
        return get_attr(self._dataset, MIN_ANGLE)

    @min_angle.setter
    def min_angle(self, value):
        set_attr(self._dataset, MIN_ANGLE, value)

    @property
    def max_angle(self):
        """
        The maximum angle value from all FFTs in this channel.
        """
        return get_attr(self._dataset, MAX_ANGLE)

    @max_angle.setter
    def max_angle(self, value):
        set_attr(self._dataset, MAX_ANGLE, value)

    @property
    def min_abs(self):
        """
        The minimum absolute value from all FFTs in this channel.
        """
        return get_attr(self._dataset, MIN_ABS)

    @min_abs.setter
    def min_abs(self, value):
        set_attr(self._dataset, MIN_ABS, value)

    @property
    def max_abs(self):
        """
        The maximum angle value from all FFTs in this channel.
        """
        return get_attr(self._dataset, MAX_ABS)

    @max_abs.setter
    def max_abs(self, value):
        set_attr(self._dataset, MAX_ABS, value)


class HDF5FFTDataSet(object):
    def __init__(self, filename : str, **kwargs):
        """
        Manages a common interface for reading and writing to HDF5 FFT dataset files.
        These files contain sets of samples that have been FFT'd and are ready for input into the GAN.

        ```python
        with HDF5FFTDataSet("fft.hdf5", "r") as fft:
            print(fft.fft_count)
        ```

        Parameters
        ----------
        filename : str
            The filename of the HDF5 file to read and write this fft dataset to.
        kwargs :
            Arguments to pass to the h5py File constructor
        """
        self.filename = filename
        self._hdf5 = h5py.File(self.filename, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._hdf5.close()
        return self

    def close(self):
        """
        Close the HDF5 file.
        """
        self._hdf5.close()

    @property
    def fft_window(self):
        """
        The size of the FFT window used to process the raw samples.
        """
        return get_attr(self._hdf5, FFT_WINDOW)

    @fft_window.setter
    def fft_window(self, value):
        set_attr(self._hdf5, FFT_WINDOW, value)

    @property
    def fft_count(self):
        """
        The total number of FFTs per channel in this file.
        """
        return get_attr(self._hdf5, FFT_COUNT)

    @fft_count.setter
    def fft_count(self, value):
        set_attr(self._hdf5, FFT_COUNT, value)

    @property
    def fft_input_size(self):
        """
        The input size of each FFT.
        """
        return get_attr(self._hdf5, FFT_INPUT_SIZE)

    @fft_input_size.setter
    def fft_input_size(self, value):
        set_attr(self._hdf5, FFT_INPUT_SIZE, value)

    @property
    def num_channels(self):
        """
        Number of channels present in the file.
        """
        return len(self._hdf5.keys())

    def get_channel(self, index: int) -> HDF5FFTChannel:
        """
        Get a specific channel from the file by index.
        Parameters
        ----------
        index : int
            The channel index to get

        Returns
        -------
        HDF5FFTChannel: The channel
        """
        name = "channel_{0}".format(index)
        return HDF5FFTChannel(name, self._hdf5[name])

    def create_channel(self, name: str, shape=None, dtype=None, data=None, **kwargs) -> HDF5FFTChannel:
        """
        Create a new channel in this FFT dataset

        Parameters
        ----------
        name: str
            The name of the channel
        shape
            The shape of the channel
        dtype
            The datatype of the channel
        data
            The channel's data
        kwargs
            Arguments to pass to the hdf5 create_dataset function

        Returns
        -------
        HDF5FFTChannel: The created channel
        """
        dataset = self.create_dataset(name, shape, dtype, data, **kwargs)
        return HDF5FFTChannel(name, dataset)

    def create_dataset(self, name: str, shape=None, dtype=None, data=None, **kwargs) -> h5py.Dataset:
        """
        Create a raw HDF5 dataset instead of a wrapped channel

        Parameters
        ----------
        name: str
            The name of the channel
        shape
            The shape of the channel
        dtype
            The datatype of the channel
        data
            The channel's data
        kwargs
            Arguments to pass to the hdf5 create_dataset function
        Returns
        -------
        h5py.Dataset: The created dataset
        """
        return self._hdf5.create_dataset(name, shape, dtype, data, **kwargs)