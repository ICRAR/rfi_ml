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

import json
import numpy as np
from preprocess_reader import PreprocessReader
from dict_validation import get_value
from lba import LBAFile


class PreprocessReaderLBA(PreprocessReader):
    """
    Preprocessor that reads an LBA file and writes to an HDF5 definition
    """

    def __init__(self, **kwargs):
        self.max_samples = get_value(kwargs, 'max_samples', types=[int], range_min=0, default_value=0)
        self.obs_filename = get_value(kwargs, 'lba_obs_file', types=[str, None], default_value=None)
        self.chunk_size = get_value(kwargs, 'chunk_size', types=[int], range_min=0, default_value=4096)

    def preprocess(self, name, input_file, observation):
        lba = LBAFile(input_file)

        if self.max_samples > 0:
            # User specified max number of samples
            max_samples = min(self.max_samples, lba.max_samples)
        else:
            max_samples = lba.max_samples

        observation.observation_name = name
        observation.original_file_name = name
        observation.original_file_type = 'lba'
        observation.additional_metadata = json.dumps(lba.header)
        observation.antenna_name = lba.header.get('ANTENNANAME', '')
        observation.sample_rate = 32000000

        samples_read = 0
        while samples_read < max_samples:
            remaining_samples = max_samples - samples_read
            samples_to_read = min(remaining_samples, self.chunk_size)
            samples = lba.read(samples_read, samples_to_read)

            for polarisation in range(samples.shape[2]):
                for channel in range(samples.shape[1]):
                    channel_name = 'p{0}_c{1}'.format(polarisation, channel)

                    out_channel = observation[channel_name]
                    if out_channel is None:
                        # storing -3, -1, 1, 3 so we can use a single byte for each
                        out_channel = observation.create_channel(channel_name, shape=(max_samples,), dtype=np.int8)
                        # out_channel.freq_start
                        # out_channel.freq_end

                    data = samples[:, channel, polarisation]
                    out_channel.write_data(samples_read, data)

            samples_read += samples_to_read
