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
Utilities for loading LBA files
"""

import mmap
import os
import struct

import numpy as np


SAMPLE_RATE = 32000000


class LBAFile(object):
    """
    Allows reading a huge LBA file using memory mapping so my IDE doesn't
    crash while trying to load 4+gb of data into ram.

    with open open('file', 'r') as f:
        lba = LBAfile(f)
        data = lba.read()
    """

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
    #      chan_def = :  6300.00 MHz : U :  16.00 MHz : &CH01 : &BBC01 : &NoCal; *Rcp
    #      chan_def = :  6642.00 MHz : U :  16.00 MHz : &CH02 : &BBC02 : &NoCal; *Rcp
    #      chan_def = :  6316.00 MHz : U :  16.00 MHz : &CH03 : &BBC01 : &NoCal; *Rcp
    #      chan_def = :  6658.00 MHz : U :  16.00 MHz : &CH04 : &BBC02 : &NoCal; *Rcp
    #      chan_def = :  6300.00 MHz : U :  16.00 MHz : &CH05 : &BBC03 : &NoCal; *Lcp
    #      chan_def = :  6642.00 MHz : U :  16.00 MHz : &CH06 : &BBC04 : &NoCal; *Lcp
    #      chan_def = :  6316.00 MHz : U :  16.00 MHz : &CH07 : &BBC03 : &NoCal; *Lcp
    #      chan_def = :  6658.00 MHz : U :  16.00 MHz : &CH08 : &BBC04 : &NoCal; *Lcp
    # enddef;
    channel_frequency_polarisation_map = [
        (0, 0),    # Chan 0
        (2, 0),    # Chan 1
        (1, 0),    # Chan 2
        (3, 0),    # Chan 3
        (0, 1),    # Chan 4
        (2, 1),    # Chan 5
        (1, 1),    # Chan 6
        (3, 1),    # Chan 7
    ]
    byte_unpack_map = {1: 'B', 2: 'H', 4: 'L', 8: 'Q'}

    def __init__(self, f):
        """
        :param f: opened file
        """
        self.mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        self.header = self._read_header()
        self.size = os.fstat(f.fileno()).st_size
        self.read_chunk_size = 4096

    @property
    def bytes_per_sample(self):
        num_chan = int(self.header["NCHAN"])
        num_bits = int(self.header["NUMBITS"])
        bandwidth = int(float(self.header["BANDWIDTH"]))

        # Each sample contains num_chan channels, the reading for each channel is num_bits
        # bandwidth >> 4 converts a bandwidth value into a byte value
        # e.g. 64 bandwidth = 4, 32 bandwidth = 2, 16 bandwidth = 1
        # final divide by 8 converts from bits to bytes
        return num_chan * num_bits * (bandwidth >> 4) // 8

    @property
    def max_samples(self):
        data_size = self.size - int(self.header["HEADERSIZE"])
        max_samples = data_size // self.bytes_per_sample
        # Skip over this number of samples, because every 32 million samples there is
        # a 65535 marker which is meaningless
        max_samples -= max_samples // 32000000
        return max_samples

    @classmethod
    def _get_frequency_polarisation(cls, channel):
        return cls.channel_frequency_polarisation_map[channel]

    def _read_header(self):
        """
        Reads in an LBA header and stores it in self.header
        It's basically just a flat key = value structure
        :return:
        """
        header = {}

        bytecount = 0
        expected_size = None  # Expected size of the header. We'll know this once we hit the "HEADERSIZE" field
        while True:
            line = self.mm.readline()
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
        :param offset: Sample index to start at (0 indexed)
        :param samples: Number of samples to read from that index.
        :return: ndarray with X = samples, Y = frequencies(4), Z = polarisations(2)
        """
        if samples < 0:
            raise Exception("Negative samples requested")

        num_chan = int(self.header["NCHAN"])
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

        # 2 polarisations per frequency, so there are half as many frequencies as channels
        # and twice as many bits per frequency.
        num_freq = num_chan // 2
        # num_freq_bits = num_bits * 2

        # Calculate number of bytes per sample
        bytes_per_sample = self.bytes_per_sample

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
        self.mm.seek(data_start + sample_offset, os.SEEK_SET)

        # X = samples, Y = frequency, Z = polarisation
        nparray = np.zeros((samples, num_freq, 2), dtype=np.int8)

        samples_output = 0  # Number of samples we dumped into nparray
        samples_read = 0  # Number of samples read, including skipped samples every 32M samples

        # Determine the data type of each sample (2 = uint16, 4 = uint32, 8 = uint64)
        struct_unpack_type = self.byte_unpack_map[bytes_per_sample]
        # Cache the structure unpack format for each read.
        # This is a string containing the format repeated for each sample read
        struct_unpack_fmt = struct_unpack_type * self.read_chunk_size
        while True:
            samples_to_read = min(samples - samples_output, self.read_chunk_size)
            if samples_to_read != self.read_chunk_size:
                # Different format for reading out the last set of bytes that might not be of the
                # chunk size cached above
                struct_unpack_fmt = struct_unpack_type * samples_to_read

            # Read a whole chunk instead of an individual sample for speed
            data_chunk = self.mm.read(samples_to_read * bytes_per_sample)
            # Unpack chunk into an array of correctly sized integers
            samples_chunk = struct.unpack(struct_unpack_fmt, data_chunk)

            for intdata in samples_chunk:
                if (samples_read + offset) % SAMPLE_RATE == 0:
                    # Richard said this was all 0s but it was actually all 1s, I hope this is correct.
                    if intdata != 65535:
                        print("Skip value should have been 65535 @ sample {0}, data may be corrupted.".format(
                            offset + samples_read
                        ))
                    else:
                        print("Skip {0} marker @ sample {1}".format(intdata, offset + samples_read))
                else:
                    for channel in range(num_chan):
                        frequency, polarisation = self.channel_frequency_polarisation_map[channel]
                        # Pull out the low two bits for P0
                        nparray[samples_output][frequency][polarisation] = \
                            val_map[intdata >> (channel * 2) & sample_mask]
                    samples_output += 1

                    if samples_output == samples:
                        break  # Got everything we need
                samples_read += 1

            if samples_output == samples:
                break

        return nparray

    def __del__(self):
        self.mm.close()
