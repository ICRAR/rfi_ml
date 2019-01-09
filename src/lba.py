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
Utilities for loading LBA files
"""

import mmap
import os
import sys
import struct
import numpy as np


class LBAFile(object):
    """
    Allows reading a huge LBA file using memory mapping so my IDE doesn't
    crash while trying to load 4+gb of data into ram.

    with open open('file', 'r') as f:
        lba = LBAfile(f)
        data = lba.read()
    """

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
        # Richard orginally gave this map [3, -3, 1, -1], but it seems to be wrong as
        # I don't get the correct spread of output values (about 2x the number of 1s as there are 3s)
        # This map was taken from some ancient csiro C code
        val_map = [3, 1, -1, -3]  # 2 bit encoding map

        # 2 polarisations per frequency, so there are half as many frequencies as channels
        # and twice as many bits per frequency.
        num_freq = num_chan // 2
        num_freq_bits = num_bits * 2

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

        # This will result in a mask for the number of bits in each frequency
        # e.g. for 4 bits per frequency, this will have the low 4 bits set
        freq_mask = (1 << num_freq_bits) - 1

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

            # data = self.mm.read(bytes_per_sample)
            # Read one sample into a byte (should be a short between 0 and 65535)
            # intdata = int.from_bytes(data, byteorder=sys.byteorder)

            for intdata in samples_chunk:
                if (samples_read + offset) % 32000000 == 0:
                    # Richard said this was all 0s but it was actually all 1s, I hope this is correct.
                    if intdata != 65535:
                        print("Skip value should have been 65535 @ sample {0}, data may be corrupted.".format(offset + samples_read))
                    else:
                        print("Skip {0} marker @ sample {1}".format(intdata, offset + samples_read))
                else:
                    for frequency in range(num_freq):
                        # One sample contains data across all frequencies (4), with two polarisations per frequency
                        # e.g. 16 bit sample: 1001,1010,0101,0000
                        # freq1: 0000, P0: 00, P1: 11
                        # freq2: 0101, P0: 01, P1: 01
                        # freq3: 1010, P0: 10, P1: 10
                        # freq4: 1001, p0: 01, p1: 10
                        freqdata = intdata >> frequency * num_freq_bits & freq_mask  # Pull out the low 4 bits for this frequency
                        nparray[samples_output][frequency][0] = val_map[freqdata & sample_mask]  # Pull out the low two bits for P0
                        nparray[samples_output][frequency][1] = val_map[freqdata >> num_bits & sample_mask]  # Pull out the high two bits for P1
                    samples_output += 1

                    if samples_output == samples:
                        break  # Got everything we need
                samples_read += 1

            if samples_output == samples:
                break

        return nparray

    def __del__(self):
        self.mm.close()
