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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from jobs import JobQueue
from lba import LBAFile
import os
import json

SAMPLE_RATE = 32000000

# Data is at 6.7GHz
# Each frequency channel is 16MHz wide (stacked upward from 6.7GHz)
# X = samples, Y = frequency band 0 to 4, Z = P0 or P1
# mlgpu, 14000M


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
    def fix_freq(f, freq_index):
        f -= np.min(f)
        f /= np.max(f)
        f *= 16.0
        f += 6700.0 + 16.0 * freq_index

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
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.bar(x, y)
        plt.title(self.get_plot_title("sample statistics histogram"))
        plt.xlabel("Sample")
        plt.ylabel("Count")
        plt.savefig(self.get_output_filename("sample_statistics_histogram.png"))
        fig.clear()
        plt.close(fig)

    def output_sample_statistics(self, samples):
        # Sample statistics
        try:
            ss = self.create_sample_statistics(samples)
            with open(self.get_output_filename("sample_statistics.json"), "w") as sf:
                json.dump(ss, sf, indent=4)
            # histogram of sample statistics
            self.save_sample_statistics_histogram(ss)
        except Exception as e:
            print("Error ouputting sample statistics {0}".format(e))

    @staticmethod
    def create_spectrogram(freq):
        return signal.spectrogram(freq, fs=SAMPLE_RATE, window=('tukey', 0.5))

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
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.xlabel("Time [sec]")
        plt.ylabel("Frequency [MHz]")
        plt.title(self.get_plot_title(title))
        plt.pcolormesh(t, f, sxx)
        plt.colorbar()
        plt.savefig(self.get_output_filename(filename))
        fig.clear()
        plt.close(fig)

    @staticmethod
    def create_periodogram(freq):
        return signal.periodogram(freq, fs=SAMPLE_RATE, window=('tukey', 0.5))

    def save_periodogram(self, periodogram):
        f, pxx = periodogram
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Power Spectral Density")
        plt.title(self.get_plot_title("periodogram"))
        plt.semilogy(f, pxx)
        plt.savefig(self.get_output_filename("periodogram"))
        fig.clear()
        plt.close(fig)

    @staticmethod
    def create_welch(freq):
        return signal.welch(freq, fs=SAMPLE_RATE, window=('tukey', 0.5))

    def save_welch(self, welch):
        f, spec = welch
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Power Spectral Density")
        plt.title(self.get_plot_title("welch"))
        plt.semilogy(f, spec)
        plt.grid()
        plt.savefig(self.get_output_filename("welch"))
        fig.clear()
        plt.close(fig)

    def create_lombscargle(self, samples):
        start = self.sample_offset / SAMPLE_RATE
        end = start + self.num_samples / SAMPLE_RATE
        times = np.linspace(start, end, samples.shape[0])

        start = 0.001
        end = 1.6e6
        freqs = np.linspace(start, end, 1000) * 10
        return freqs, signal.lombscargle(times, samples, freqs, normalize=True)

    def save_lombscargle(self, lombscargle):
        f, pgram = lombscargle
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Power Spectral Density")
        plt.title(self.get_plot_title("lombscargle"))
        plt.plot(f, pgram)
        plt.savefig(self.get_output_filename("lombscargle"))
        fig.clear()
        plt.close(fig)

    def create_fft(self, samples):
        ft = np.fft.rfft(samples * signal.windows.tukey(samples.shape[0], sym=False))
        f = np.fft.rfftfreq(samples.shape[0], d=SAMPLE_RATE)
        np.abs(ft, ft)
        return f, ft

    def save_fft(self, fft):
        f, ft = fft
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Power")
        plt.title(self.get_plot_title("fft"))
        plt.plot(f, ft)
        plt.savefig(self.get_output_filename("fft"))
        fig.clear()
        plt.close(fig)

    def create_psd(self, samples):
        return mlab.psd(samples, Fs=SAMPLE_RATE, window=signal.get_window(('tukey', 0.5), 256))

    def save_psd_asd(self, psd, type):
        Pxx, freqs = psd
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel(type)
        plt.title(self.get_plot_title(type))
        plt.plot(freqs, Pxx)
        plt.savefig(self.get_output_filename(type))
        fig.clear()
        plt.close(fig)

    def __call__(self):
        # Firstly, create the output directory
        os.makedirs(self.out_directory, exist_ok=True)

        if self.filename.endswith(".lba"):
            with open(self.filename, "r") as f:
                lba = LBAFile(f)
                samples = lba.read(self.sample_offset, self.num_samples)
        elif self.filename.endswith(".npz"):
            samples = np.load(self.filename)["arr_0"]

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

                # Spectrogram for this frequency
                f, t, sxx = self.create_spectrogram(freq_samples)
                # Calculate the actual frequencies for the spectrogram
                self.fix_freq(f, freq)
                spectrogram = (f, t, sxx)
                spectrograms.append(spectrogram)
                self.save_spectrogram(spectrogram)

                # Periodogram for this frequency
                f, pxx = self.create_periodogram(freq_samples)
                self.fix_freq(f, freq)
                self.save_periodogram((f, pxx))

                # Welch
                f, spec = self.create_welch(freq_samples)
                self.fix_freq(f, freq)
                self.save_welch((f, spec))

                # Lombscargle
                try:
                    f, pxx = self.create_lombscargle(freq_samples)
                    self.fix_freq(f, freq)
                    self.save_lombscargle((f, pxx))
                except ZeroDivisionError:
                    print("Zero division in Lombscargle")

                # FFT
                f, ft = self.create_fft(freq_samples)
                self.fix_freq(f, freq)
                self.save_fft((f, ft))

                # power spectral density
                Pxx, f = self.create_psd(freq_samples)
                self.fix_freq(f, freq)
                self.save_psd_asd((Pxx, f), "psd")

                # amplitude spectral density
                np.sqrt(Pxx, Pxx)
                self.save_psd_asd((Pxx, f), "asd")

            self.frequency = None

            # Create merged spectrograms for this p
            merged = self.merge_spectrograms(spectrograms)
            merged_normalised = self.merge_spectrograms(spectrograms, normalise_local=True)
            self.save_spectrogram(merged, "merged", "spectrogram_merged.png")
            self.save_spectrogram(merged_normalised, "merged local normalisation", "spectrogram_merged_normalised.png")


if __name__ == "__main__":
    queue = JobQueue(8)

    # Load each file using a process pool
    num_samples = SAMPLE_RATE # should be 1 second
    queue.submit(LBAPlotter("../data/v255ae_At_072_060000.lba", "./At_out/", num_samples=num_samples))
    queue.submit(LBAPlotter("../data/v255ae_Mp_072_060000.lba", "./Mp_out/", num_samples=num_samples))
    queue.submit(LBAPlotter("../data/vt255ae_Pa_072_060000.lba", "./Pa_out/", num_samples=num_samples))
    #queue.submit(LBAPlotter("../data/downsamples.npz", "./downsamples_out/"))

    queue.join()
