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
Preprocess an india powerline text file into an `src.preprocess.hdf5_definition.HDF5Observation` file.
"""

import os
import json
import logging
import itertools
import datetime
import numpy as np

from .hdf5_definition import HDF5Observation
from .preprocess_reader import PreprocessReader

LOG = logging.getLogger(__name__)


def _grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.
    ```python
    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    ```

    Parameters
    ----------
    iterable
    n
    fillvalue

    Returns
    -------

    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


class PreprocessReaderIndiaTXT(PreprocessReader):

    _write_cache_size = 1024 * 1024
    _header_types = {
        'Segments': int,
        'SegmentSize': int,
        'Segment': int,
        'TrigTime': lambda x: datetime.datetime.strptime(x, '%d-%b-%Y %H:%M:%S').timestamp(),
        'TimeSinceSegment1': int
    }

    def __init__(self, **kwargs):
        """
        Preprocess an India powerline text file.

        Parameters
        ----------
        kwargs
            Keyword arguments for the processor. See `src.preprocess.main.parse_args`.
        """
        self.sample_rate = kwargs.get('sample_rate', 200000000)  # 200MHz
        if self.sample_rate is None:
            self.sample_rate = 200000000

    @classmethod
    def _read_header(cls, input_file):
        header = {}

        def update(new_dict):
            for k, v in new_dict:
                converter = cls._header_types.get(k, None)
                header[k] = converter(v) if converter is not None else v

        # First line contains some random things
        line_split = input_file.readline().strip().split(',')
        update([(k, '') for k in line_split])

        # Second line also contains random things, but they seem to have a kv mapping
        # between each set of 2 elements
        line_split = input_file.readline().strip().split(',')
        update(_grouper(line_split, 2))

        # Next two lines seem to map to each other
        line_split1 = input_file.readline().strip().split(',')
        line_split2 = input_file.readline().strip().strip("#").split(',')
        update(zip(line_split1, line_split2))

        # Final line should be Time,Ampl, just skip it
        input_file.readline()

        return header

    def preprocess(self, name: str, input_file, observation: HDF5Observation):
        """
        Preprocess an india powerline file.

        Parameters
        ----------
        name : str
            The file name to process
        input_file : file
            The input file open for rading
        observation : HDF5Observation
            The observation file to write to
        """
        header = self._read_header(input_file)

        num_samples = int(header["SegmentSize"])

        observation.observation_name = os.path.basename(name)
        observation.length_seconds = num_samples / self.sample_rate
        observation.sample_rate = self.sample_rate
        observation.num_channels = 1
        observation.additional_metadata = json.dumps(header)
        observation.num_samples = num_samples

        channel = observation.create_channel("channel_0", (num_samples,), dtype=np.float32)
        channel.write_defaults()

        out_array = np.zeros(shape=(self._write_cache_size,), dtype=np.float32)
        cache_count = 0
        write_index = 0
        first_time = None
        for line in input_file:
            line_split = line.split(',')
            out_array[cache_count] = float(line_split[1])
            cache_count += 1

            if first_time is None:
                first_time = float(line_split[0])

            if cache_count == self._write_cache_size:
                LOG.info("Writing out {0} samples at index {1}".format(out_array.shape[0], write_index))
                channel.write_data(write_index, out_array)
                cache_count = 0
                write_index += self._write_cache_size

        if cache_count > 0:
            channel.write_data(write_index, out_array[:cache_count])

        # Work out the actual start time by adding the time of the first sample
        # to the observation time
        observation.start_time = header["TrigTime"] + first_time

        LOG.info("India TXT preprocessor complete")
