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

import gc
import json
import logging
import os

import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal

from lba import LBAFile

LOGGER = logging.getLogger(__name__)
SAMPLE_RATE = 32000000


# | Station Modes | At Mp Pa                                |
# |---------------|-----------------------------------------|
# | Channel 1	  | DAS #1 IFP#1-LO 6300 - 6316 MHz USB RCP |
# | Channel 2	  | DAS #1 IFP#1-HI 6316 - 6332 MHz USB RCP |
# | Channel 3	  | DAS #1 IFP#2-LO 6300 - 6316 MHz USB LCP |
# | Channel 4	  | DAS #1 IFP#2-HI 6316 - 6332 MHz USB LCP |
# | Channel 5	  | DAS #2 IFP#1-LO 6642 - 6658 MHz USB RCP |
# | Channel 6	  | DAS #2 IFP#1-HI 6658 - 6674 MHz USB RCP |
# | Channel 7	  | DAS #2 IFP#2-LO 6642 - 6658 MHz USB LCP |
# | Channel 8	  | DAS #2 IFP#2-HI 6658 - 6674 MHz USB LCP |
# | DAS 1 Skyfreq | 6316 MHz                                |
# | DAS 2 Skyfreq | 6658 MHz                                |
# | Bandwidth	  | 16 MHz                                  |
#
# Each frequency channel is 16MHz wide (stacked upward from 6.3GHz and 6.642GHz)


class LBAPlotter(object):
    # Frequency range for each channel within a polarisation
    # http://www.atnf.csiro.au/vlbi/dokuwiki/doku.php/lbaops/lbamar2018/v255ae
    channel_frequency_map = [
        (6300, 6316),  # f1
        (6316, 6332),  # f2
        (6642, 6658),  # f3
        (6658, 6674)   # f4
    ]

    def __init__(self, filename, out_directory, sample_offset=0, num_samples_=0):
        self.filename = filename
        self.basefilename = os.path.basename(self.filename)
        self.out_directory = out_directory
        self.sample_offset = sample_offset
        self.num_samples = num_samples_
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

    @classmethod
    def fix_freq(cls, f, freq_index):
        (freq_start, freq_end) = cls.channel_frequency_map[freq_index]
        f -= np.min(f)
        f /= np.max(f)
        f *= (freq_end - freq_start)
        f += freq_start

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
        with PdfPages(self.get_output_filename("sample_statistics_histogram.pdf")) as pdf:
            plt.figure(figsize=(16, 9))
            plt.bar(x, y)
            plt.title(self.get_plot_title("sample statistics histogram"))
            plt.xlabel("Sample")
            plt.ylabel("Count")

            plt.tight_layout()
            pdf.savefig()
            plt.close()

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

    @staticmethod
    def merge_spectrograms(spectrograms, normalise_local=False):
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

    def save_spectrogram(self, spectogram, title="spectrogram", filename="spectrogram.pdf"):
        f, t, sxx = spectogram
        with PdfPages(self.get_output_filename(filename)) as pdf:
            plt.figure(figsize=(16, 9))
            plt.xlabel("Time [sec]")
            plt.ylabel("Frequency [MHz]")
            plt.title(self.get_plot_title(title))
            plt.pcolormesh(t, f, sxx)
            plt.colorbar()

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    @staticmethod
    def create_periodogram(freq):
        return signal.periodogram(freq, fs=SAMPLE_RATE, window=('tukey', 0.5))

    def save_periodogram(self, periodogram):
        f, pxx = periodogram
        with PdfPages(self.get_output_filename("periodogram.pdf")) as pdf:
            plt.figure(figsize=(16, 9))
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Power Spectral Density [V/rtMHz]")
            plt.title(self.get_plot_title("periodogram"))
            plt.grid()
            plt.plot(f, pxx)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    @staticmethod
    def create_welch(freq):
        return signal.welch(freq, fs=SAMPLE_RATE, window=('tukey', 0.5))

    def save_welch(self, welch):
        f, spec = welch
        with PdfPages(self.get_output_filename("welch.pdf")) as pdf:
            plt.figure(figsize=(16, 9))
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Power Spectral Density [V/rtMHz]")
            plt.title(self.get_plot_title("welch"))
            plt.grid()
            plt.plot(f, spec)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

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
        with PdfPages(self.get_output_filename("lombscargle.pdf")) as pdf:
            plt.figure(figsize=(16, 9))
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Power Spectral Density [V/rtMHz]")
            plt.title(self.get_plot_title("lombscargle"))
            plt.grid()
            plt.plot(f, pgram)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    @staticmethod
    def create_rfft(samples):
        ft = np.fft.rfft(samples * signal.windows.tukey(samples.shape[0], sym=False))
        f = np.fft.rfftfreq(samples.shape[0], d=SAMPLE_RATE)
        np.abs(ft, ft)
        return f, ft

    def save_rfft(self, fft):
        f, ft = fft
        with PdfPages(self.get_output_filename("fft.pdf")) as pdf:
            plt.figure(figsize=(16, 9))
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Power [V]")
            plt.title(self.get_plot_title("fft"))
            plt.grid()
            plt.plot(f, ft)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    @staticmethod
    def create_ifft(samples):
        ft = np.fft.fft(samples * signal.windows.tukey(samples.shape[0]))
        f = np.fft.fftfreq(samples.shape[0], d=SAMPLE_RATE)
        return f, np.imag(ft)

    def save_ifft(self, fft):
        f, ft = fft
        with PdfPages(self.get_output_filename("ifft.pdf")) as pdf:
            plt.figure(figsize=(16, 9))
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Power [V]")
            plt.title(self.get_plot_title("ifft"))
            plt.grid()
            plt.plot(f, ft)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    @staticmethod
    def create_psd(samples):
        return mlab.psd(samples, Fs=SAMPLE_RATE, window=signal.get_window(('tukey', 0.5), 256))

    def save_psd(self, psd, type, ylabel):
        pxx, freqs = psd
        with PdfPages(self.get_output_filename(type + '.pdf')) as pdf:
            plt.figure(figsize=(16, 9))
            plt.xlabel("Frequency [MHz]")
            plt.ylabel(ylabel)
            plt.title(self.get_plot_title(type))
            plt.grid()
            plt.plot(freqs, pxx)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    def __call__(self):

        LOGGER.info("Plotter for {0} started".format(self.filename))
        matplotlib.rc('font', weight='normal', size=18)

        # Firstly, create the output directory
        os.makedirs(self.out_directory, exist_ok=True)
        LOGGER.info("Output directory {0} created".format(self.out_directory))

        if self.filename.endswith(".lba"):
            with open(self.filename, "r") as f:
                LOGGER.info("Opening LBA file {0} and reading samples...".format(self.filename))
                lba = LBAFile(f)
                samples = lba.read(self.sample_offset, self.num_samples)
                del lba
                gc.collect()  # Ensure the loaded lba file is unloaded
        elif self.filename.endswith(".npz"):
            samples = np.load(self.filename)["arr_0"]
        else:
            raise ValueError('{} not recognised'.format(self.filename))

        LOGGER.info("Read {0} samples".format(self.num_samples))

        # Do global things across all samples
        LOGGER.info("Calculating sample statistics for entire dataset...")
        self.output_sample_statistics(samples)

        # Iterate over each of the two polarisations
        for pindex in range(samples.shape[2]):
            LOGGER.info("{0} Polarisation {1}".format(self.filename, pindex))
            p = samples[:, :, pindex]

            self.polarisation = pindex
            os.makedirs(self.get_output_filename(), exist_ok=True)
            LOGGER.info("{0}, P{1} Sample statistics...".format(self.filename, pindex))
            self.output_sample_statistics(p)

            spectrogram_groups = [[], []]  # freq1, freq2 : freq3, freq4
            # Iterate over each of the four frequencies
            for freq in range(p.shape[1]):
                LOGGER.info("{0}, P{1} Frequency {2}".format(self.filename, pindex, freq))
                freq_samples = p[:, freq]

                self.frequency = freq
                os.makedirs(self.get_output_filename(), exist_ok=True)
                LOGGER.info("{0}, P{1}, F{2} Sample statistics...".format(self.filename, pindex, freq))
                self.output_sample_statistics(freq_samples)

                # Spectrogram for this frequency
                LOGGER.info("{0}, P{1}, F{2} Spectrogram".format(self.filename, pindex, freq))
                f, t, sxx = self.create_spectrogram(freq_samples)
                # Calculate the actual frequencies for the spectrogram
                self.fix_freq(f, freq)
                spectrogram = (f, t, sxx)
                spectrogram_groups[freq // 2].append(spectrogram)
                LOGGER.info("{0}, P{1}, F{2} Spectrogram Saving".format(self.filename, pindex, freq))
                self.save_spectrogram(spectrogram)

                # Periodogram for this frequency
                LOGGER.info("{0}, P{1}, F{2} Periodogram".format(self.filename, pindex, freq))
                f, pxx = self.create_periodogram(freq_samples)
                self.fix_freq(f, freq)
                LOGGER.info("{0}, P{1}, F{2} Periodogram Saving".format(self.filename, pindex, freq))
                self.save_periodogram((f, pxx))

                # Welch
                LOGGER.info("{0}, P{1}, F{2} Welch".format(self.filename, pindex, freq))
                f, spec = self.create_welch(freq_samples)
                self.fix_freq(f, freq)
                LOGGER.info("{0}, P{1}, F{2} Welch Saving".format(self.filename, pindex, freq))
                self.save_welch((f, spec))

                # Lombscargle
                try:
                    LOGGER.info("{0}, P{1}, F{2} Lombscargle".format(self.filename, pindex, freq))
                    f, pxx = self.create_lombscargle(freq_samples)
                    self.fix_freq(f, freq)
                    LOGGER.info("{0}, P{1}, F{2} Lombscargle Saving".format(self.filename, pindex, freq))
                    self.save_lombscargle((f, pxx))
                except ZeroDivisionError:
                    print("Zero division in Lombscargle")

                # RFFT
                LOGGER.info("{0}, P{1}, F{2} RFFT".format(self.filename, pindex, freq))
                f, ft = self.create_rfft(freq_samples)
                self.fix_freq(f, freq)
                LOGGER.info("{0}, P{1}, F{2} RFFT Saving".format(self.filename, pindex, freq))
                self.save_rfft((f, ft))

                # IFFT
                LOGGER.info("{0}, P{1}, F{2} IRFFT".format(self.filename, pindex, freq))
                f, ft = self.create_ifft(freq_samples)
                self.fix_freq(f, freq)
                LOGGER.info("{0}, P{1}, F{2} IRFFT Saving".format(self.filename, pindex, freq))
                self.save_ifft((f, ft))

                # power spectral density
                LOGGER.info("{0}, P{1}, F{2} PSD".format(self.filename, pindex, freq))
                pxx, f = self.create_psd(freq_samples)
                self.fix_freq(f, freq)
                LOGGER.info("{0}, P{1}, F{2} PSD Saving".format(self.filename, pindex, freq))
                self.save_psd((pxx, f), "Power Spectral Density", "Power Spectral Density [V/rtMHz]")

                # amplitude spectral density
                LOGGER.info("{0}, P{1}, F{2} ASD".format(self.filename, pindex, freq))
                np.sqrt(pxx, pxx)
                LOGGER.info("{0}, P{1}, F{2} ASD Saving".format(self.filename, pindex, freq))
                self.save_psd((pxx, f), "Amplitude Spectral Density", "Amplitude Spectral Density [sqrt(V/rtMHz)]")

            self.frequency = None

            # Create merged spectrograms for this p
            LOGGER.info("{0}, P{1} Merged Spectrograms...".format(self.filename, pindex))
            for index, group in enumerate(spectrogram_groups):
                LOGGER.info("{0}, P{1} Merged Spectrograms Group {2}".format(self.filename, pindex, index))
                merged = self.merge_spectrograms(group)
                merged_normalised = self.merge_spectrograms(group, normalise_local=True)
                LOGGER.info("{0}, P{1} Merged Spectrograms Group {2} Saving".format(self.filename, pindex, index))
                self.save_spectrogram(
                    merged, 
                    "group {0} merged".format(index), 
                    "spectrogram_group{0}_merged.pdf".format(index)
                )
                self.save_spectrogram(
                    merged_normalised, 
                    "group {0} merged local normalisation".format(index), 
                    "spectrogram_group{0}_merged_normalised.pdf".format(index)
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.INFO)

    num_samples = 1919999940 // 2048  # SAMPLE_RATE # should be 1 second
    LBAPlotter("../../data/v255ae_At_072_060000.lba", "./At_out/", num_samples_=num_samples)()
    gc.collect()
    LBAPlotter("../../data/v255ae_Mp_072_060000.lba", "./Mp_out/", num_samples_=num_samples)()
    gc.collect()
    LBAPlotter("../../data/vt255ae_Pa_072_060000.lba", "./Pa_out/", num_samples_=num_samples)()
    gc.collect()
