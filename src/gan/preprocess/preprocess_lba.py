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

import os
import json
import numpy as np
import pyvex
import logging
import datetime
import collections
from astropy import time
from preprocess_reader import PreprocessReader
from dict_validation import get_value
from lba import LBAFile

LOG = logging.getLogger(__name__)

Scan = collections.namedtuple("Scan", "start stop")


class PreprocessReaderLBA(PreprocessReader):
    """
    Preprocessor that reads an LBA file and writes to an HDF5 definition
    """

    default_sample_rate = 32000000

    def __init__(self, **kwargs):
        self.max_samples = get_value(kwargs, 'max_samples', types=[int], range_min=0, default_value=0)
        self.obs_filename = get_value(kwargs, 'lba_obs_file', types=[str, None], default_value=None)
        self.sample_rate = get_value(kwargs, 'sample_rate', types=[int, None], default_value=None)
        if self.sample_rate is None:
            LOG.warning("No sample rate provided, defaulting to {0}".format(self.default_sample_rate))
            self.sample_rate = self.default_sample_rate
        self.antenna_name = get_value(kwargs, 'lba_antenna_name', types=[str, None], default_value=None)
        if self.obs_filename is not None and self.antenna_name is None:
            raise RuntimeError("LBA file is missing --lba_antenna_name parameter which is needed when using "
                               "--lba_obs_file")
        self.chunk_size = get_value(kwargs, 'chunk_size', types=[int], range_min=0, default_value=4096)

    def _fill_source_array(self, observation, vex, start, end, max_samples):
        start_mjd = time.Time(start).mjd
        original_start = start_mjd
        end_mjd = time.Time(end).mjd
        total_delta = end_mjd - start_mjd
        # Ensure scans are sorted per start time
        # Get the scans for this specific antenna

        antenna_name = next(a.name for a in vex.antennas if a.def_name == self.antenna_name)

        scans = filter(lambda s: antenna_name in s.stations, vex.scans)
        scans = map(lambda s: s.stations[antenna_name], scans)
        scans = sorted(scans, key=lambda s: s.start)
        sources = []

        # TODO: This needs to use the correct scans for the specific station that we're reading
        #       data for. The start and end times will vary for the station..

        for scan in scans:
            if start_mjd >= end_mjd:
                break  # Can't fit any more scans in

            if scan.start > start_mjd:
                # Next scan begins after our current start period
                # Advance the start period
                start_mjd = min(scan.start, end_mjd)

            if scan.start <= start_mjd:
                end = min(scan.stop, end_mjd)
                duration = end - start_mjd
                fill_start = int(((start_mjd - original_start) / total_delta) * max_samples)
                fill_size = int((duration / total_delta) * max_samples)
                sources.append([fill_start, fill_start + fill_size])
                start_mjd = end

        observation.create_dataset('sources', data=np.array(sources), dtype=np.uint64)

    def preprocess(self, name, input_file, observation):

        lba = LBAFile(input_file, self.sample_rate)

        if self.max_samples > 0:
            # User specified max number of samples
            max_samples = min(self.max_samples, lba.max_samples)
        else:
            max_samples = lba.max_samples

        # Length of the observation for the number of samples we're reading
        obs_length = lba.obs_length(max_samples)
        start = lba.obs_start
        end = start + datetime.timedelta(seconds=obs_length)

        LOG.info("LBA obs time: start {0} end {1} duration {2} sec".format(start, end, obs_length))

        observation.observation_name = os.path.basename(name)
        observation.additional_metadata = json.dumps(lba.header)
        observation.antenna_name = self.antenna_name if self.antenna_name is not None else lba.header.get('ANTENNANAME', '')
        observation.sample_rate = self.sample_rate
        observation.length_seconds = obs_length
        observation.start_time = start.timestamp()
        observation.num_channels = lba.num_channels
        observation.num_samples = max_samples

        channel_map = None
        if self.obs_filename is not None:
            try:
                vex = pyvex.Vex(self.obs_filename)
                self._fill_source_array(observation, vex, start, end, max_samples)

                # Get channel info from the VEX file.
                # Pick the appropriate mode from the vex file. We run with the assumption for now that there's only one
                # mode, and get everything from it. If there are multiple modes, then error out
                if len(vex.modes) == 0:
                    LOG.warning("No modes in vex file to get channel info from")
                elif len(vex.modes) > 1:
                    LOG.error("Cannot get channel information from vex file because multiple modes are present. This is currently unsupported")
                else:
                    # Get antenna info
                    antenna = next((a for a in vex.antennas if a.def_name == self.antenna_name), None)
                    if antenna is None:
                        LOG.error("Specified antenna def name {0} is not present in the vex file".format(self.antenna_name))
                    else:
                        LOG.info("Found antenna def name {0}. Name {1}".format(self.antenna_name, antenna.name))
                        mode = vex.modes[0]
                        setup = mode.setups[antenna.name]
                        channel_map = [[channel, mode.subbands[channel.subband_id], setup.ifs["IF_{0}".format(channel.if_name)]] for channel in setup.channels]
                        # TODO: I'm unsure if this is correct or what order we should use here
                        #       - Ordering by subband_id will match the CH0(0-8) ordering present in the vex file.
                        #       - Ordering by record_chan seems to make some logic sense as it orders them from low to
                        #       high frequency across pol R then pol L. It also orders the BBC values ascending.
                        #       - Neither of these seem to match the observation_details.md table.

                        # TODO: For now, I'll assume sorting by record chan
                        channel_map.sort(key=lambda c: c[0].record_chan)

            except Exception as e:
                LOG.error("Failed to parse vex file {0}".format(e))

        samples_read = 0
        while samples_read < max_samples:
            remaining_samples = max_samples - samples_read
            samples_to_read = min(remaining_samples, self.chunk_size)
            samples = lba.read(samples_read, samples_to_read)

            for channel_index in range(samples.shape[1]):
                # Loop over each channel, then either output the channel directly
                # into the HDF5 file, or get the channel info to get the appropriate metadata
                # for the channel, then output that.

                channel_name = "channel_{0}".format(channel_index)

                out_channel = observation[channel_name]
                if out_channel is None:
                    out_channel = observation.create_channel(channel_name, shape=(max_samples,), dtype=np.int8)
                    if channel_map is not None:
                        vex_channel, vex_subband, vex_if = channel_map[channel_index]
                        # Add info from the vex file to the channel
                        out_channel.freq_start = vex_channel.bbc_freq
                        out_channel.freq_end = vex_channel.bbc_freq + vex_channel.bbc_bandwidth
                        out_channel.additional_metadata = json.dumps({
                            'channel_name': vex_channel.name,
                            'polarisation': vex_subband.pol.decode('utf-8'),
                            'bbc_name': vex_channel.bbc_name,
                            'record_chan': vex_channel.record_chan,
                            'subband_id': vex_channel.subband_id,
                            'vlba_band_name': vex_if.vlba_band_name,
                            'if_sslo': vex_if.if_sslo,
                            'lower_edge_freq': vex_if.lower_edge_freq
                        }, indent=4)
                data = samples[:, channel_index]
                out_channel.write_data(samples_read, data)

            samples_read += samples_to_read

            if (samples_read // self.chunk_size) % 10 == 0:
                LOG.info("{0:.2%} {1}/{2}".format(samples_read / max_samples, samples_read, max_samples))

        LOG.info("LBA preprocessor complete")
