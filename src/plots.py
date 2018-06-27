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
Methods to generate a variety of plots from lba files
"""

import cupy as cp
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from jobs import JobQueue
from lba import LBAFile
import os
import json

SAMPLE_RATE = 32000000

# Data is at 6.7GHz
# Each frequency channel is 16MHz wide (stacked upward from 6.7GHz)
# X = samples, Y = frequency band 0 to 4, Z = P0 or P1
# mlgpu, 14000M

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


class LBAPlotter(object):
    def __init__(self, filename, out_directory, sample_offset=0, num_samples=0):
        self.filename = filename
        self.basefilename = os.path.basename(self.filename)
        self.out_directory = out_directory
        self.sample_offset = sample_offset
        self.num_samples = num_samples
        self.polarisation = None
        self.frequency = None

    def get_output_filename(self, filename=""):
        path = self.out_directory
        if self.polarisation is not None:
            path = os.path.join(path, "p{0}".format(self.polarisation))
            if self.frequency is not None:
                path = os.path.join(path, "f{0}".format(self.frequency))
        return os.path.join(path, filename)

    def get_plot_title(self, plot_title):
        title = os.path.splitext(self.basefilename)[0]
        if self.polarisation is not None:
            title += " p{0}".format(self.polarisation)
            if self.frequency is not None:
                title += " f{0}".format(self.frequency)
        return "{0} {1}".format(title, plot_title)

    @staticmethod
    def create_sample_statistics(samples):
        """
        Prints some counting statistics for the provided samples
        :param samples:
        :return:
        """
        unique, counts = np.unique(samples, return_counts=True)
        counts = dict(zip([int(x) for x in unique], [int(x) for x in counts]))
        negative = counts[-3] + counts[-1]
        positive = counts[3] + counts[1]
        low = counts[-1] + counts[1]
        high = counts[-3] + counts[3]

        return {
            "shape": samples.shape,
            "counts": counts,
            "neg_counts": negative,
            "pos_counts": positive,
            "neg_pos_ratio": negative / positive,
            "low": low,
            "high": high,
            "low_high_radio": low / high
        }

    def save_sample_statistics_histogram(self, sample_statistics):
        """
        Saves a histogram of sample statistics
        :param sample_statistics:
        :return:
        """
        x = [-3, -1, 1, 3]
        y = [sample_statistics["counts"][i] for i in x]
        plt.bar(x, y)
        plt.title(self.get_plot_title("sample statistics histogram"))
        plt.xlabel("Sample")
        plt.ylabel("Count")
        plt.savefig(self.get_output_filename("sample_statistics_histogram.png"))
        plt.gcf().clear()

    def output_sample_statistics(self, samples):
        # Sample statistics
        ss = self.create_sample_statistics(samples)
        with open(self.get_output_filename("sample_statistics.txt"), "w") as sf:
            json.dump(ss, sf, indent=4)
        # histogram of sample statistics
        self.save_sample_statistics_histogram(ss)

    @staticmethod
    def create_spectrogram(freq):
        return signal.spectrogram(freq, fs=SAMPLE_RATE)

    def merge_spectrograms(self, spectrograms, normalise_local=False):
        fs = []
        ts = spectrograms[0][1]
        sxxs = []
        for freq_index, (f, t, sxx) in enumerate(spectrograms):
            if normalise_local:
                mins = np.min(sxx)
                maxs = np.max(sxx)
                sxx -= mins
                sxx /= maxs
            fs.append(f)
            sxxs.append(sxx)

        return np.concatenate(fs), ts, np.concatenate(sxxs)

    def save_spectrogram(self, spectogram, title="spectrogram", filename="spectrogram.png"):
        f, t, sxx = spectogram
        plt.figure(figsize=(16, 9), dpi=80)
        plt.xlabel("Time [sec]")
        plt.ylabel("Frequency [MHz]")
        plt.title(self.get_plot_title(title))
        plt.pcolormesh(t, f, sxx)
        plt.colorbar()
        plt.savefig(self.get_output_filename(filename))
        plt.gcf().clear()

    def __call__(self):
        # Firstly, create the output directory
        os.makedirs(self.out_directory, exist_ok=True)

        with open(self.filename, "r") as f:
            lba = LBAFile(f)
            samples = lba.read(self.sample_offset, self.num_samples)

            # Do global things across all samples
            self.output_sample_statistics(samples)

            # Split into p0 and p1
            p0 = samples[:, :, 0]
            p1 = samples[:, :, 1]
            for pindex, p in enumerate((p0, p1)):
                # Do things for each polarisation
                self.polarisation = pindex
                os.makedirs(self.get_output_filename(), exist_ok=True)

                self.output_sample_statistics(p)

                spectrograms = []

                for freq in range(p.shape[1]):
                    # Do things for each frequency
                    freq_samples = p[:, freq]

                    self.frequency = freq
                    os.makedirs(self.get_output_filename(), exist_ok=True)

                    self.output_sample_statistics(freq_samples)
                    f, t, sxx = self.create_spectrogram(freq_samples)
                    # Calculate the actual frequencies for the spectrogram
                    f /= 1e6
                    f += 6700 + 16 * freq
                    spectrogram = (f, t, sxx)
                    spectrograms.append(spectrogram)
                    self.save_spectrogram(spectrogram)

                self.frequency = None

                # Create merged spectrograms for this p
                merged = self.merge_spectrograms(spectrograms)
                merged_normalised = self.merge_spectrograms(spectrograms, normalise_local=True)
                self.save_spectrogram(merged, "merged", "spectrogram_merged.png")
                self.save_spectrogram(merged_normalised, "merged local normalisation", "spectrogram_merged_normalised.png")


if __name__ == "__main__":
    queue = JobQueue(8)

    # Load each file using a process pool
    num_samples = 102400
    queue.submit(LBAPlotter("../data/v255ae_At_072_060000.lba", "./At_out/", num_samples=num_samples))
    queue.submit(LBAPlotter("../data/v255ae_Mp_072_060000.lba", "./Mp_out/", num_samples=num_samples))
    queue.submit(LBAPlotter("../data/vt255ae_Pa_072_060000.lba", "./Pa_out/", num_samples=num_samples))

    queue.join()
