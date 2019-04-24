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
Preprocessing pipeline step 1: Extract data from an input file (lba etc.) and write to a common HDF5 format.
HDF5 format contains the following information
Metadata
- Observation name (if any)
- Start time (unix epoch)
- Length (ms)
- Sample rate
- Original file name
- Original file type
- Additional (string of any additional metadata)
Channels
- Group for each channel
    Metadata
    - Channel name (if any)
    - Antenna name (if any)
    - Frequency start
    - Frequency end
    - Additional (string of any additional metadata)
    Data
    - One continuous array of voltage samples
    - Timestamp, (ra, dec) for all available timestamps
    - Timestamp, Event ID, Event Details (Can be used to store 'on source', 'off source' etc. information)
        -Valid event IDs:
            0: ON_SOURCE
            1: OFF_SOURCE
"""

import h5py
from hdf5_utils import Attribute, get_attr, set_attr

# HDF5 root attributes
OBSERVATION_NAME = Attribute('observation_name', str)
ANTENNA_NAME = Attribute('antenna_name', str)
START_TIME = Attribute('start_time_posix', int)
LENGTH_SECONDS = Attribute('length_seconds', float)
SAMPLE_RATE = Attribute('sample_rate_hz', int)
FILE_NAME = Attribute('file_name', str)
FILE_TYPE = Attribute('file_type', str)
NUM_CHANNELS = Attribute('num_channels', str)

# HDF5 dataset attributes
FREQ_START = Attribute('freq_start_mhz', float)
FREQ_END = Attribute('freq_end_mhz', float)

# Shared attributes (used in both root and dataset)
ADDITIONAL_METADATA = Attribute('additional_metadata', str)


class HDF5Channel(object):
    """
    Manages a common interface for writing channels to HDF5 datasets to meet the spec.
    """
    def __init__(self, name, dataset):
        self._name = name
        self._dataset = dataset

    def write_data(self, offset, data):
        self._dataset[offset:offset + data.shape[0]] = data

    def read_data(self, offset, length):
        return self._dataset[offset:offset + length]

    @property
    def name(self):
        """
        Get the name of this channel
        :return:
        """
        return self._name

    @property
    def freq_start(self):
        """
        Get the start freq of this channel in MHz
        :return:
        """
        return get_attr(self._dataset, FREQ_START)

    @freq_start.setter
    def freq_start(self, freq):
        """
        Set the start freq of this channel in MHz
        :param freq:
        :return:
        """
        set_attr(self._dataset, FREQ_START, freq)

    @property
    def freq_end(self):
        """
        Get the end freq of this channel in MHz
        :return:
        """
        return get_attr(self._dataset, FREQ_END)

    @freq_end.setter
    def freq_end(self, freq):
        """
        Set the end freq of this channel in MHz
        :param freq:
        :return:
        """
        set_attr(self._dataset, FREQ_END, freq)

    @property
    def additional_metadata(self):
        """
        Get the additional metadata associated with the channel.
        :return:
        """
        return get_attr(self._dataset, ADDITIONAL_METADATA)

    @additional_metadata.setter
    def additional_metadata(self, metadata):
        """
        Set the metadata associated with the observation.
        :param metadata:
        :return:
        """
        set_attr(self._dataset, ADDITIONAL_METADATA, metadata)


class HDF5Observation(object):
    """
    Manages a common interface for reading and writing to HDF5 files to meet the spec.
    """

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self._hdf5 = h5py.File(self.filename, mode='w')
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

    def create_channel(self, name, shape=None, dtype=None, data=None, **kwargs):
        dataset = self.create_dataset(name, shape, dtype, data, **kwargs)
        return HDF5Channel(name, dataset)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwargs):
        return self._hdf5.create_dataset(name, shape, dtype, data, **kwargs)

    @property
    def observation_name(self):
        """
        Get the observation name
        :return:
        """
        return get_attr(self._hdf5, OBSERVATION_NAME)

    @observation_name.setter
    def observation_name(self, name):
        """
        Set the observation name
        :param name:
        :return:
        """
        set_attr(self._hdf5, OBSERVATION_NAME, name)

    @property
    def antenna_name(self):
        """
        Get the antenna name of this channel
        :return:
        """
        return get_attr(self._hdf5, ANTENNA_NAME)

    @antenna_name.setter
    def antenna_name(self, name):
        """
        Set the antenna name of this channel
        :param name:
        :return:
        """
        set_attr(self._hdf5, ANTENNA_NAME, name)

    @property
    def start_time(self):
        """
        Get observation start time as a unix epoch in milliseconds
        :return:
        """
        return get_attr(self._hdf5, START_TIME)

    @start_time.setter
    def start_time(self, time):
        """
        Set observation the start time as a unix epoch in milliseconds
        :param time:
        :return:
        """
        set_attr(self._hdf5, START_TIME, time)

    @property
    def length_seconds(self):
        """
        Get the duration of the observation in seconds
        :return:
        """
        return get_attr(self._hdf5, LENGTH_SECONDS)

    @length_seconds.setter
    def length_seconds(self, length):
        """
        Set the duration of the observation in seconds
        :param length:
        :return:
        """
        set_attr(self._hdf5, LENGTH_SECONDS, length)

    @property
    def sample_rate(self):
        """
        Get the sample rate in samples per second
        :return:
        """
        return get_attr(self._hdf5, SAMPLE_RATE)

    @sample_rate.setter
    def sample_rate(self, rate):
        """
        Set the sample rate in samples per second
        :param rate:
        :return:
        """
        set_attr(self._hdf5, SAMPLE_RATE, rate)

    @property
    def original_file_name(self):
        """
        Set the original name of the file that contained the observation,
        before the file was converted to HDF5.
        :return:
        """
        return get_attr(self._hdf5, FILE_NAME)

    @original_file_name.setter
    def original_file_name(self, name):
        """
        Set the original name of the file that contained the observation.
        :param name:
        :return:
        """
        set_attr(self._hdf5, FILE_NAME, name)

    @property
    def original_file_type(self):
        """
        Get the original type of the file that contained the observation.
        :return:
        """
        return get_attr(self._hdf5, FILE_TYPE)

    @original_file_type.setter
    def original_file_type(self, t):
        """
        Set the original type of the file that contained the observation.
        :param t:
        :return:
        """
        set_attr(self._hdf5, FILE_TYPE, t)

    @property
    def num_channels(self):
        """
        Get the number of channels in the HDF5 file
        :return:
        """
        return get_attr(self._hdf5, NUM_CHANNELS)

    @num_channels.setter
    def num_channels(self, t):
        """
        Set the number of channels in the HDF5 file
        :param t:
        :return:
        """
        set_attr(self._hdf5, NUM_CHANNELS, t)

    @property
    def additional_metadata(self):
        """
        Get the additional metadata associated with the observation.
        :return:
        """
        return get_attr(self._hdf5, ADDITIONAL_METADATA)

    @additional_metadata.setter
    def additional_metadata(self, metadata):
        """
        Set the metadata associated with the observation.
        :param metadata:
        :return:
        """
        set_attr(self._hdf5, ADDITIONAL_METADATA, metadata)
