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
Generate gaussian noise and plot it
"""
import numpy as np
from lba import LBAFile
from scipy import signal
from matplotlib import pyplot as plt

SAMPLE_RATE = 32000000


def main():
    for i in range(500000):
        noise = np.random.normal(-0.0289923828125, 1.9391296947313124, 262144)
        fft = np.abs(np.fft.rfft(noise))[10:-10]

        indexes = np.argwhere(fft > np.mean(fft) + np.std(fft) * 6)
        if indexes.shape[0] > 2:
            fig = plt.figure(figsize=(16, 9), dpi=80)
            plt.plot(fft)
            plt.plot(indexes, fft[indexes], 'x')
            plt.show()
            fig.clear()
            plt.close(fig)

            f, t, sxx = signal.spectrogram(noise, fs=SAMPLE_RATE, window=('tukey', 0.5))
            fig = plt.figure(figsize=(16, 9), dpi=80)
            plt.xlabel("Time [sec]")
            plt.ylabel("Frequency [MHz]")
            plt.pcolormesh(t, f, sxx)
            plt.colorbar()
            plt.show()
            fig.clear()
            plt.close(fig)


def mean_stddev():
    with open("/home/sam/Projects/rfi_ml/data/v255ae_At_072_060000.lba", 'r') as f:
        lba = LBAFile(f)
        samples = lba.read(0, 6400000).flatten()
        mean = np.mean(samples)
        stddev = np.std(samples)
        print(mean, stddev)


if __name__ == "__main__":
    #mean_stddev()
    main()