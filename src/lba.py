# -*- coding: utf-8 -*-
#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2018
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
Class for loading .lba files as numpy arrays.
"""

import mmap
import os
import struct
import re
import datetime
import logging

import numpy as np

LOG = logging.getLogger(__name__)


class LBAFile(object):
    _date_regex = re.compile(r"(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2}):"
                            r"(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})")

    # For the Numpy array holding the data
    # Freq
    # 0 = 6300-6316
    # 1 = 6316-6332
    # 2 = 6642-6658
    # 3 = 6658-6674
    #
    # Pol
    # 0 = RCP
    # 1 = LCP
    #
    # $FREQ;
    # *
    # def 6300.00MHz8x16MHz;
    # * mode =  1    stations =At:Mp:Pa
    #      sample_rate =  32.000 Ms/sec;  * (2bits/sample)
    # | Channel 1	    | DAS #1 IFP#1-LO 6300 - 6316 MHz USB RCP |
    # | Channel 2	    | DAS #1 IFP#1-HI 6316 - 6332 MHz USB RCP |
    # | Channel 3	    | DAS #1 IFP#2-LO 6300 - 6316 MHz USB LCP |
    # | Channel 4	    | DAS #1 IFP#2-HI 6316 - 6332 MHz USB LCP |
    # | Channel 5	    | DAS #2 IFP#1-LO 6642 - 6658 MHz USB RCP |
    # | Channel 6	    | DAS #2 IFP#1-HI 6658 - 6674 MHz USB RCP |
    # | Channel 7	    | DAS #2 IFP#2-LO 6642 - 6658 MHz USB LCP |
    # | Channel 8	    | DAS #2 IFP#2-HI 6658 - 6674 MHz USB LCP |
    # enddef;

    _byte_unpack_map = {1: 'B', 2: 'H', 4: 'L', 8: 'Q'}

    def __init__(self, f, sample_rate):
        """
        Load an LBA file to read samples from it as numpy arrays.
        ```python
        with open open('file', 'r') as f:
            lba = LBAfile(f)
            data = lba.read()
        ```
        Parameters
        ----------
        f : file
            The open file to read from
        sample_rate : int
            The sample rate of the data in the file, in Hz
        """
        self._mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        self.header = self._read_header()
        """
        dict: The LBA header file. A dict containing key-value parsed pased from the LBA header.
        """
        self._size = os.fstat(f.fileno()).st_size
        self._sample_rate = sample_rate
        self._read_chunk_size = 4096

    @property
    def num_channels(self) -> int:
        """
        int: Count of the number of channels stored within the LBA file.
        """
        return int(self.header["NCHAN"])

    @property
    def _bytes_per_sample(self):
        num_chan = self.num_channels
        num_bits = int(self.header["NUMBITS"])
        bandwidth = int(float(self.header["BANDWIDTH"]))

        # Each sample contains num_chan channels, the reading for each channel is num_bits
        # bandwidth >> 4 converts a bandwidth value into a byte value
        # e.g. 64 bandwidth = 4, 32 bandwidth = 2, 16 bandwidth = 1
        # final divide by 8 converts from bits to bytes
        return num_chan * num_bits * (bandwidth >> 4) // 8

    @property
    def max_samples(self):
        """
        int: Maximum number of samples that can be read from the LBA file.
        """
        data_size = self._size - int(self.header["HEADERSIZE"])
        max_samples = data_size // self._bytes_per_sample
        return max_samples

    @property
    def obs_start(self):
        """
        datetime: Get the time that this observation started, from the TIME header value.
        """
        matches = self._date_regex.match(self.header["TIME"])
        if matches is None:
            return None
        else:
            return datetime.datetime(
                int(matches.group("year")),
                int(matches.group("month")),
                int(matches.group("day")),
                int(matches.group("hour")),
                int(matches.group("minute")),
                int(matches.group("second"))
            )

    def obs_length(self, samples=None):
        """
        Determine how long part or all of this LBA file is, in seconds.
        This is based on the sample rate.

        Parameters
        ----------
        samples : int
            If provided, determine how long this many samples is, in seconds.
            If not provided, determine how long the entire LBA file is, in seconds.

        Returns
        -------
        int: Length of the provided number of samples, in seconds.
        """
        if samples is None:
            samples = self.max_samples
        return samples / self._sample_rate

    def _read_header(self):
        """
        Reads in an LBA header and stores it in self.header
        It's basically just a flat key = value structure
        """
        header = {}

        bytecount = 0
        expected_size = None  # Expected size of the header. We'll know this once we hit the "HEADERSIZE" field
        while True:
            line = self._mm.readline()
            if line is None:
                break

            bytecount += len(line)
            if expected_size is not None and bytecount >= expected_size:
                break  # Gone over expected size of header

            line = line.strip()

            if line == b"END":
                break  # Hit the end of header flag

            k, v = line.split(b' ', 1)
            header_key = k.decode("utf-8")
            header_value = v.decode("utf-8")
            header[header_key] = header_value

            if header_key == "HEADERSIZE":
                expected_size = int(header_value)

        return header

    def read(self, offset=0, samples=0):
        """
        Reads a set of samples out of the lba file.
        Note that you can read samples from anywhere in the file by specifying
        an offset to start at.

        Parameters
        ----------
        offset : int
            Sample index to start at (0 indexed)
        samples : int
            Number of samples to read from that index.
        Returns
        -------
        ndarray: with X = samples, Y = channels
        Raises
        -------
        Exception
            In any of the following cases:

            - samples < 0
            - offset < 0
            - samples > max_samples
            - offset > max_samples
            - offset + samples > max_samples
        """
        if samples < 0:
            raise Exception("Negative samples requested")

        num_chan = self.num_channels
        num_bits = int(self.header["NUMBITS"])
        data_start = int(self.header["HEADERSIZE"])

        # Richard originally gave this map [3, -3, 1, -1], but it seems to be wrong as
        # I don't get the correct spread of output values (about 2x the number of 1s as there are 3s)
        # This map was taken from some ancient csiro C code
        #
        # 23/02/19: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20110011794.pdf page 196
        # VDIF-encoded data samples are represented by the desired number of bits in a fixed-point
        # ‘offset binary sequence’, beginning with all 0’s for the most-negative sampled value to all 1’s for
        # the most-positive sampled value. For example, 2-bit/sample coding is (in order from most negative
        # to most positive) 00, 01, 10, 11. This coding is compatible with existing Mark 5B, K5, and LBADR
        # disk-based VLBI data systems, though bit-ordering may be different in some cases
        #
        val_map = [3, 1, -1, -3]  # 2 bit encoding map

        # Calculate number of bytes per sample
        bytes_per_sample = self._bytes_per_sample

        # Max samples that can be requested from the file
        max_samples = self.max_samples
        if samples == 0:
            samples = max_samples
        elif samples > max_samples:
            raise Exception("{0} samples requested with {1} max samples".format(samples, max_samples))

        # Confirm that the user requested a sane offset
        if offset > max_samples:
            raise Exception("Offset {0} > Maxsamples {1}".format(offset, max_samples))
        elif offset < 0:
            raise Exception("Offset {0} < 0".format(offset))

        if offset + samples > max_samples:
            raise Exception("Offset {0}, samples {1} will overflow lba file".format(offset, samples))

        sample_offset = offset * bytes_per_sample

        # This will result in a mask for the number of bits in a single sample
        # e.g. for 2 bits per sample, this will have the low 2 bits set
        sample_mask = (1 << num_bits) - 1

        # Seek to the desired offset
        self._mm.seek(data_start + sample_offset, os.SEEK_SET)

        # X = samples, Y = channels.
        nparray = np.zeros((samples, num_chan), dtype=np.int8)

        samples_output = 0  # Number of samples we dumped into nparray
        samples_read = 0  # Number of samples read, including skipped samples every 32M samples

        # Determine the data type of each sample (1 = uint8, 2 = uint16, 4 = uint32, 8 = uint64)
        struct_unpack_type = self._byte_unpack_map[bytes_per_sample]
        # Cache the structure unpack format for each read.
        # This is a string containing the format repeated for each sample read
        struct_unpack_fmt = struct_unpack_type * self._read_chunk_size
        while True:
            samples_to_read = min(samples - samples_output, self._read_chunk_size)
            if samples_to_read != self._read_chunk_size:
                # Different format for reading out the last set of bytes that might not be of the
                # chunk size cached above
                struct_unpack_fmt = struct_unpack_type * samples_to_read

            # Read a whole chunk instead of an individual sample for speed
            data_chunk = self._mm.read(samples_to_read * bytes_per_sample)
            # Unpack chunk into an array of correctly sized integers
            samples_chunk = struct.unpack(struct_unpack_fmt, data_chunk)

            for intdata in samples_chunk:
                # This now just checks if this is the case but doesn't skip it
                if (samples_read + offset) % self._sample_rate == 0:
                    # Richard said this was all 0s but it was actually all 1s, I hope this is correct.
                    if intdata != 65535:
                        LOG.info("Skip value should have been 65535 at sample {0}. Data may be corrupted.".format(
                            offset + samples_read
                        ))
                    else:
                        LOG.info("Skip {0} marker at sample {1}. Data validity check passed".format(
                            intdata, offset + samples_read
                        ))

                for channel in range(num_chan):
                    # Changed to simply return the channels as a flat array from 0 - num_chan.
                    # The actual info associated with the channels is derived from the vex file
                    # or the channels are simply passed through as 0 - num_chan.
                    # This work is done in the preprocess_lba.py script
                    nparray[samples_output][channel] = val_map[intdata >> (channel * 2) & sample_mask]
                samples_output += 1

                if samples_output == samples:
                    break  # Got everything we need
                samples_read += 1

            if samples_output == samples:
                break

        return nparray

    def __del__(self):
        self._mm.close()
