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
Defines the spec for an `src.preprocess.hdf5_definition.HDF5Observation` file.
"""

import h5py
import numpy as np

from .hdf5_utils import Attribute, get_attr, set_attr

# HDF5 root attributes
OBSERVATION_NAME = Attribute('observation_name', str)
ANTENNA_NAME = Attribute('antenna_name', str)
START_TIME = Attribute('start_time_posix', float)
LENGTH_SECONDS = Attribute('length_seconds', float)
SAMPLE_RATE = Attribute('sample_rate_hz', int)
FILE_NAME = Attribute('file_name', str)
FILE_TYPE = Attribute('file_type', str)
NUM_CHANNELS = Attribute('num_channels', int)
NUM_SAMPLES = Attribute('num_samples', int)

# HDF5 dataset attributes
FREQ_START = Attribute('freq_start_mhz', float)
FREQ_END = Attribute('freq_end_mhz', float)

# Shared attributes (used in both root and dataset)
ADDITIONAL_METADATA = Attribute('additional_metadata', str)


class HDF5Channel(object):
    def __init__(self, name: str, dataset: h5py.Dataset):
        """
        Manages a common interface for writing channels to HDF5 datasets to meet the spec.

        Parameters
        ----------
        name : str
            The name of the channel
        dataset : h5py.Dataset
            The dataset that corresponds to this channel
        """
        self._name = name
        self._dataset = dataset

    def write_defaults(self):
        """
        Write default metadata values to the channel.
        """
        self.freq_start = 0
        self.freq_end = 0
        self.additional_metadata = ""

    def write_data(self, offset: int, data: np.ndarray):
        """
        Write data into this channel

        Parameters
        ----------
        offset : int
            Offset in the channel to write data to.
        data : np.ndarray
            Data to write
        """
        self._dataset[offset:offset + data.shape[0]] = data

    def read_data(self, offset: int, length: int) -> np.ndarray:
        """
        Read data from the channel

        Parameters
        ----------
        offset : int
            Offset in the channel to read from.
        length : int
            Length of data to read from the channel

        Returns
        -------
        ndarray: Numpy array containing the data.
        """
        return self._dataset[offset:offset + length]

    @property
    def name(self) -> str:
        """
        The name of this channel
        """
        return self._name

    @property
    def length(self) -> int:
        """
        The length of the dataset
        """
        return self._dataset.shape[0]

    @property
    def freq_start(self):
        """
        The start freq of this channel in MHz
        """
        return get_attr(self._dataset, FREQ_START)

    @freq_start.setter
    def freq_start(self, freq):
        set_attr(self._dataset, FREQ_START, freq)

    @property
    def freq_end(self):
        """
        The end freq of this channel in MHz
        """
        return get_attr(self._dataset, FREQ_END)

    @freq_end.setter
    def freq_end(self, freq):
        set_attr(self._dataset, FREQ_END, freq)

    @property
    def additional_metadata(self):
        """
        Additional metadata associated with the channel.
        """
        return get_attr(self._dataset, ADDITIONAL_METADATA)

    @additional_metadata.setter
    def additional_metadata(self, metadata):
        set_attr(self._dataset, ADDITIONAL_METADATA, metadata)


class HDF5Observation(object):

    def __init__(self, filename: str, **kwargs):
        """
        Manages a common interface for reading and writing to HDF5 observation files.
        These files contain raw samples parsed by the preprocessor.

        ```python
        with HDF5Observation("obs.hdf5", "r") as hdf5:
            print(hdf5.observation_name)
        ```

        Parameters
        ----------
        filename : str
            The filename of the HDF5 file to read and write this observation to.
        kwargs :
            Arguments to pass to the h5py File constructor
        """
        self.filename = filename
        self.kwargs = kwargs

    def __enter__(self):
        self._hdf5 = h5py.File(self.filename, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._hdf5.close()
        return self

    def __contains__(self, item):
        return item in self._hdf5

    def __getitem__(self, item):
        dataset = self._hdf5.get(item)
        if dataset is None:
            return None
        return HDF5Channel(item, dataset)

    def close(self):
        """
        Close the HDF5 file.
        """
        self._hdf5.close()

    def write_defaults(self):
        """
        Write default metadata to the HDF5 file.
        """
        self.observation_name = ""
        self.antenna_name = ""
        self.start_time = 0
        self.length_seconds = 0
        self.sample_rate = 0
        self.original_file_name = ""
        self.original_file_type = ""
        self.num_channels = 0
        self.additional_metadata = ""

    def create_channel(self, name: str, shape=None, dtype=None, data=None, **kwargs) -> HDF5Channel:
        """
        Create a channel in the HDF5 file.

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
        HDF5Channel: The created channel
        """
        dataset = self.create_dataset(name, shape, dtype, data, **kwargs)
        return HDF5Channel(name, dataset)

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

    @property
    def observation_name(self):
        """
        The observation name
        """
        return get_attr(self._hdf5, OBSERVATION_NAME)

    @observation_name.setter
    def observation_name(self, name):
        set_attr(self._hdf5, OBSERVATION_NAME, name)

    @property
    def antenna_name(self):
        """
        The antenna name
        """
        return get_attr(self._hdf5, ANTENNA_NAME)

    @antenna_name.setter
    def antenna_name(self, name):
        set_attr(self._hdf5, ANTENNA_NAME, name)

    @property
    def start_time(self):
        """
        The observation start time as a unix epoch in milliseconds
        """
        return get_attr(self._hdf5, START_TIME)

    @start_time.setter
    def start_time(self, time):
        set_attr(self._hdf5, START_TIME, time)

    @property
    def length_seconds(self):
        """
        The duration of the observation in seconds
        """
        return get_attr(self._hdf5, LENGTH_SECONDS)

    @length_seconds.setter
    def length_seconds(self, length):
        set_attr(self._hdf5, LENGTH_SECONDS, length)

    @property
    def sample_rate(self):
        """
        The sample rate in samples per second
        """
        return get_attr(self._hdf5, SAMPLE_RATE)

    @sample_rate.setter
    def sample_rate(self, rate):
        set_attr(self._hdf5, SAMPLE_RATE, rate)

    @property
    def original_file_name(self):
        """
        The original name of the file that contained the observation,
        before the file was converted to HDF5.
        """
        return get_attr(self._hdf5, FILE_NAME)

    @original_file_name.setter
    def original_file_name(self, name):
        set_attr(self._hdf5, FILE_NAME, name)

    @property
    def original_file_type(self):
        """
        The original type of the file that contained the observation.
        """
        return get_attr(self._hdf5, FILE_TYPE)

    @original_file_type.setter
    def original_file_type(self, t):
        set_attr(self._hdf5, FILE_TYPE, t)

    @property
    def num_channels(self):
        """
        The number of channels in the HDF5 file
        """
        return get_attr(self._hdf5, NUM_CHANNELS)

    @num_channels.setter
    def num_channels(self, t):
        set_attr(self._hdf5, NUM_CHANNELS, t)

    @property
    def num_samples(self):
        """
        The number of samples stored in each channel in the HDF5 file.
        Each channel should have the same number of samples.
        """
        return get_attr(self._hdf5, NUM_SAMPLES)

    @num_samples.setter
    def num_samples(self, value):
        set_attr(self._hdf5, NUM_SAMPLES, value)

    @property
    def additional_metadata(self):
        """
        Additional metadata associated with the observation.
        """
        return get_attr(self._hdf5, ADDITIONAL_METADATA)

    @additional_metadata.setter
    def additional_metadata(self, metadata):
        set_attr(self._hdf5, ADDITIONAL_METADATA, metadata)
