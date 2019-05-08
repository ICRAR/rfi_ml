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
from hdf5_utils import Attribute, get_attr, set_attr

MIN_ANGLE = Attribute('min_angle', float)
MAX_ANGLE = Attribute('max_angle', float)
MIN_ABS = Attribute('min_abs', float)
MAX_ABS = Attribute('max_abs', float)


class HDF5FFTChannel(object):
    def __init__(self, name, dataset):
        self._name = name
        self._dataset = dataset

    def write_data(self, offset, data):
        """
        Write FFT data to the channel
        :param offset: Offset in the channel to put the data
        :param data: 2D numpy array of FFT values. x = fft index, y = fft values
                     e.g. (2, 128) would be 2 ffts of 128 length
        """
        self._dataset[offset:offset + data.shape[0]] = data

    def read_data(self, offset, count):
        """
        Read one or more FFTs from the channel
        :param offset: Offset to start reading
        :param count: Number of FFTs to read
        :return: Numpy array containing as many ffts as are available up to the desired count
        """
        return self._dataset[offset:offset + count]

    @property
    def min_angle(self):
        return get_attr(self._dataset, MIN_ANGLE)

    @min_angle.setter
    def min_angle(self, value):
        set_attr(self._dataset, MIN_ANGLE, value)

    @property
    def max_angle(self):
        return get_attr(self._dataset, MAX_ANGLE)

    @max_angle.setter
    def max_angle(self, value):
        set_attr(self._dataset, MAX_ANGLE, value)

    @property
    def min_abs(self):
        return get_attr(self._dataset, MIN_ABS)

    @min_abs.setter
    def min_abs(self, value):
        set_attr(self._dataset, MIN_ABS, value)

    @property
    def max_abs(self):
        return get_attr(self._dataset, MAX_ABS)

    @max_abs.setter
    def max_abs(self, value):
        set_attr(self._dataset, MAX_ABS, value)


class HDF5FFTDataSet(object):
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self._hdf5 = h5py.File(self.filename, mode='w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._hdf5.close()
        return self

    def create_channel(self, name, shape=None, dtype=None, data=None, **kwargs):
        dataset = self.create_dataset(name, shape, dtype, data, **kwargs)
        return HDF5FFTChannel(name, dataset)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwargs):
        return self._hdf5.create_dataset(name, shape, dtype, data, **kwargs)