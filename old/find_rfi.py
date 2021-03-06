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
Try and find RFI within the LBA files
"""

import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from lba import LBAFile
from scipy import signal, stats

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)

SAMPLE_RATE = 32000000


def rfi_mean_stddev(samples, findex, pindex, read_index, out_filename):
    # fft = (np.abs(np.fft.rfft(samples[:, findex, pindex]))[10:-10] ** 2) / sample_window
    ft, fft = signal.welch(samples[:, findex, pindex], SAMPLE_RATE)
    ft = ft[10:-10]
    fft = fft[10:-10]

    # Ignore the lower and upper frequencies as they often contain trash
    # fft = np.abs(np.fft.rfft(samples[:, findex, pindex]))[10:-10]
    indexes = np.argwhere(fft > np.mean(fft) + np.std(fft) * 6)

    if indexes.shape[0] > 2 or read_index == 0:
        # Found peaks, plot them
        location = "p{0} f{1} s{2}".format(pindex, findex, read_index)
        LOG.info("Found peak: {0}".format(location))
        plot_peaks(fft, indexes, "{0}_{1}.png".format(out_filename, location))


def rfi_kurtosis(samples, findex, pindex, read_index, out_filename):
    f, t, Zxx = signal.stft(samples[:, findex, pindex], fs=SAMPLE_RATE)
    k = stats.kurtosis(np.abs(Zxx))
    m = np.mean(np.abs(Zxx))
    print("Kurtosis of {0} f{1} p{2}: {3}, mean: {4}".format(read_index, findex, pindex, k, m))


def plot_peaks(fft, peaks, filename):
    fig = plt.figure(figsize=(16, 9), dpi=80)
    plt.plot(fft)
    plt.plot(peaks, fft[peaks], 'x')
    plt.savefig(filename)
    fig.clear()
    plt.close(fig)


def search_rfi(filename, out_filename, sample_window, start_sample):
    with open(filename, 'r') as f:
        lba = LBAFile(f)
        read_index = start_sample

        while read_index < lba.max_samples:
            samples = lba.read(read_index, min(sample_window, lba.max_samples - read_index))
            LOG.info("Index: {0}".format(read_index))

            for pindex in range(samples.shape[2]):
                for findex in range(samples.shape[1]):
                    #rfi_mean_stddev(samples, findex, pindex, read_index, out_filename)
                    rfi_kurtosis(samples, findex, pindex, read_index, out_filename)

            read_index += sample_window


def parse_args():
    parser = argparse.ArgumentParser(description="Search an LBA file for RFI")
    parser.add_argument('lba_file', type=str, help="LBA file to search")
    parser.add_argument('out_file', type=str, help="File to output potential found RFI sources")
    parser.add_argument('sample_window', type=int, help="Number of samples per window")
    parser.add_argument('start_sample', type=int, default=0, help="Sample to start searching at")
    return vars(parser.parse_args())


def main():
    args = parse_args()
    search_rfi(args['lba_file'], args['out_file'], args['sample_window'], args['start_sample'])


if __name__ == "__main__":
    main()
