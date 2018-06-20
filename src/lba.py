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
import math
import os
import sys
import numpy as np
import cupy as cp
from scipy import signal
import matplotlib.pyplot as plt


class LBAFile(object):
    """
    Allows reading a huge LBA file using memory mapping so my IDE doesn't
    crash while trying to load 4+gb of data into ram.

    with open open('file', 'r') as f:
        lba = LBAfile(f)
        data = lba.read()
    """

    def __init__(self, f):
        """
        :param f: opened file
        """
        self.mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        self.header = self._read_header()
        self.size = os.fstat(f.fileno()).st_size

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

        bandwidth = int(float(self.header["BANDWIDTH"]))
        num_chan = int(self.header["NCHAN"])
        num_bits = int(self.header["NUMBITS"])
        data_size = self.size - int(self.header["HEADERSIZE"])
        data_start = int(self.header["HEADERSIZE"])
        # Richard orginally gave this map [3, -3, 1, -1], but it seems to be wrong as
        # I don't get the correct spread of output values (about 2x the number of 1s as there are 3s)
        # This map was taken from some ancient csiro C code
        val_map = [3, 1, -1, -3]  # 2 bit encoding map

        # 2 polarisations per frequency, so there are half as many frequencies as channels
        # and twice as many bits per frequency.
        num_freq = num_chan // 2
        num_freq_bits = num_bits * 2

        # Each sample contains num_chan channels, the reading for each channel is num_bits
        # bandwidth >> 4 converts a bandwidth value into a byte value
        # e.g. 64 bandwidth = 4, 32 bandwidth = 2, 16 bandwidth = 1
        # final divide by 8 converts from bits to bytes
        bytes_per_sample = num_chan * num_bits * (bandwidth >> 4) // 8

        # Confirm that the user requested a sane amount of data.
        max_samples = data_size // bytes_per_sample
        # Skip over this number of samples, because every 32 million samples there is
        # a 65535 marker which is meaningless
        max_samples -= max_samples // 32000000
        if samples == 0:
            samples = max_samples
        elif samples > max_samples:
            raise Exception("{0} samples requested with {1} max samples".format(samples, max_samples))

        # Confirm that the user requested a sane offset
        sample_offset = offset * bytes_per_sample
        if sample_offset > max_samples:
            raise Exception("Offset {0} > Maxsamples {1}".format(sample_offset, max_samples))
        elif sample_offset < 0:
            raise Exception("Offset {0} < 0".format(sample_offset))

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
        while True:
            data = self.mm.read(bytes_per_sample)
            # Read one sample into a byte (should be a short between 0 and 65535)
            intdata = int.from_bytes(data, byteorder=sys.byteorder)

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

        return nparray

    def __del__(self):
        self.mm.close()


def print_sample_stats(samples):
    """
    Prints some counting statistics for the provided samples
    :param samples:
    :return:
    """
    unique, counts = cp.unique(samples, return_counts=True)
    counts = dict(zip(unique, counts))
    print("Shape {0}".format(samples.shape))
    print("Counts {0}".format(counts))
    negative = counts[-3.0] + counts[-1.0]
    positive = counts[3.0] + counts[1.0]
    print("Negative count {0}".format(negative))
    print("Positive count {0}".format(positive))
    print("Negative to positive ratio {0}".format(negative / positive))
    low = counts[-1.0] + counts[1.0]
    high = counts[-3.0] + counts[3.0]
    print("Low count {0}".format(low))
    print("High count {0}".format(high))
    print("Low to high ratio {0}".format(low / high))


def show_histogram(samples, title):
    """
    Shows a histogram of polarisations found in the provided samples
    :param samples:
    :param title:
    :return:
    """
    plt.hist(cp.reshape(samples, samples.shape[0] * 4))
    plt.title(title)
    plt.xlabel("polarisation")
    plt.ylabel("count")
    plt.show()


def fft_freq_sort(samples):
    """
    Uses cp.fft.fftfreq to generate sample frequencies.
    Frequencies and samples are sorted from lowest to highest frequency to ensure
    results are plotted correctly.
    :param samples:
    :return:
    """
    freq_sample = cp.fft.fftfreq(samples.shape[0], d=32000000)
    sort_indices = cp.argsort(freq_sample)
    return freq_sample[sort_indices], samples[sort_indices]


def fft_linspace(samples):
    """
    Generates sample 'frequencies' starting at 0, going to num_samples / 2
    This should show half the fft result
    :param samples:
    :return:
    """
    freq_sample = cp.linspace(0, 1.0 / (2.0 * 32000000), samples.shape[0] // 2)
    return freq_sample, samples[0:samples.shape[0] // 2]


def fft_sum_amplitudes(samples):
    """
    Calculates the amplitude abs(ft) for all frequencies, and sums them, returning
    the summed amplitudes
    :param samples:
    :return:
    """
    # Entire file contains 1920000000 samples over 1 minute
    # 32000000 samples per second
    # Fourier transform and sum amplitudes
    # fftfreq provides x axis values, amplitude is y axis values
    amps = None #cp.zeros(samples.shape[0])
    for freq in range(samples.shape[1]):
        ft = cp.abs(cp.fft.rfft(samples[:, freq]))
        if amps is None:
            amps = ft
        else :
            amps += ft

    return amps


def fft_sum_amplitudes2(samples):
    """
    Sums amplitudes in a different way, including some sort of normalisation.
    I found it here and thought I'd experiment:
    https://docs.scipy.org/doc/scipy-1.1.0/reference/tutorial/fftpack.html#one-dimensional-discrete-fourier-transforms
    :param samples:
    :return:
    """
    amps = []
    for freq in range(samples.shape[1]):
        ft = cp.fft.fft(samples[:, freq])
        amps.append(2.0 / samples.shape[0] * cp.abs(ft))
    return cp.sum(amps, axis=0)


def fft_power_spectrum(samples):
    """
    Pretty sure this calculates the power spectrum. That's what the equation at
    https://www.wavemetrics.com/products/igorpro/dataanalysis/signalprocessing/powerspectra.htm
    seems to say.
    Also sums it up but I'm not sure if I'm meant to do that
    :param samples:
    :return:
    """
    sum = []
    for freq in range(samples.shape[1]):
        ft = cp.fft.fft(samples[:, freq])
        sum.append((cp.abs(ft)**2) / samples.shape[0])

    return cp.sum(sum, axis=0)


def create_spectrogram(p):
    def handle_tuple(t):
        return cp.array(t[0]), t[1], cp.array(t[2])

    return [handle_tuple(signal.spectrogram(cp.asnumpy(p[:, freq]), fs=32000000)) for freq in range(p.shape[1])]


def merge_spectrograms(spectrograms, start_frequency, bandwidth, normalise_local=False):
    fs = cp.zeros((len(spectrograms), spectrograms[0][0].shape[0]))
    sxxs = cp.zeros((len(spectrograms), spectrograms[0][2].shape[0], spectrograms[0][2].shape[1]))
    for freq_index, (f, t, sxx) in enumerate(spectrograms):
        f = (f / cp.max(f)) * bandwidth + start_frequency + freq_index * bandwidth
        fs[freq_index] = f
        if normalise_local:
            sxx = (sxx - cp.min(sxx)) / cp.max(sxx)
        sxxs[freq_index] = sxx

    return cp.concatenate(fs), spectrograms[0][1], cp.concatenate(sxxs)


def show_spectrogram(spectogram, title):
    f, t, sxx = spectogram
    plt.figure(figsize=(20, 10), dpi=100)
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [MHz]")
    plt.title(title)
    plt.pcolormesh(cp.asnumpy(t), cp.asnumpy(f), cp.asnumpy(sxx))
    plt.colorbar()
    plt.savefig("{0}.png".format(title))
    #plt.show()


def plot_spectrograms(polarisation, filename, pindex):
    s = create_spectrogram(polarisation)
    for sindex, (f, t, sxx) in enumerate(s):
        show_spectrogram((f, t, sxx), "P{0} {1} channel {2}".format(pindex, filename, sindex))
    f, t, sxx = merge_spectrograms(s, 6700, 16, normalise_local=True)
    show_spectrogram((f, t, sxx), "P{0} {1} local normalised test".format(pindex, filename))

    f, t, sxx = merge_spectrograms(s, 6700, 16, normalise_local=False)
    show_spectrogram((f, t, sxx), "P{0} {1} non normalised test".format(pindex, filename))


def plot_fft(polarisation, filename, pindex):
    x, y = fft_freq_sort(fft_sum_amplitudes(polarisation))
    plt.title("P{0} {1} fft".format(pindex, filename))
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")

    #x, y = fft_linspace(fft_sum_amplitudes2(polarisation))
    plt.plot(cp.asnumpy(x), cp.asnumpy(y), label="p{0}".format(pindex), lw=0.5)
    plt.legend()
    plt.show()


def plot_over_time(polarisation, filename, pindex):
    for freq in range(polarisation.shape[1]):
        y = polarisation[:, freq]
        x = cp.linspace(0, y.shape[0] / 32000000.0, y.shape[0])
        plt.title("P{0} {1} over time".format(pindex, filename))
        plt.xlabel("Time"),
        plt.ylabel("Power")
        plt.plot(x, y)
        plt.show()


def run_main(filename):
    """
    Runs a bunch of tests on the provided file
    :param filename:
    :return:
    """
    with open(filename, "r") as f:
        lba = LBAFile(f)
        print("\n", lba.header)

        num_samples = 102400
        # Data is at 6.7GHz
        # Each frequency channel is 16MHz wide (stacked upward from 6.7GHz)
        # X = samples, Y = frequency band 0 to 4, Z = P0 or P1
        # mlgpu, 14000M
        print("Reading samples...")
        samples = lba.read(samples=num_samples)
        # Split up into P0 and P1
        # X = samples, Y = P0 for frequency band 0 to 4
        all_p0 = cp.array(samples[:, :, 0])
        # X = samples, Y = P1 for frequency band 0 to 4
        all_p1 = cp.array(samples[:, :, 1])

        basename = os.path.basename(filename)

        print("Creating plots...")
        for index, polarisation in enumerate((all_p0, all_p1)):
            print("Plotting FFT")
            plot_fft(polarisation, basename, index)
            print("Plotting spectrogram")
            plot_spectrograms(polarisation, basename, index)
            #plot_over_time(polarisation, basename, index)



if __name__ == "__main__":
    run_main("../data/v255ae_At_072_060000.lba")
    # run_main("../data/v255ae_Mp_072_060000.lba")
    # run_main("../data/vt255ae_Pa_072_060000.lba")
