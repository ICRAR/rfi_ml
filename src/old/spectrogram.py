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
Generate a spectrogram across a part of an lba file
"""

import os
import argparse
import logging
import matplotlib.pyplot as plt
from lba import LBAFile
from scipy import signal

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


SAMPLE_RATE = 32000000


def create_spectrograms(filename, out_filename, start_sample, num_samples):
    with open(filename, 'r') as f:
        lba = LBAFile(f)
        samples = lba.read(start_sample, num_samples)
        LOG.info("Read {0} samples".format(num_samples))
        os.makedirs("{0}_{1}".format(out_filename, start_sample), exist_ok=True)

        for pindex in range(samples.shape[2]):
            for findex in range(samples.shape[1]):
                f, t, sxx = signal.spectrogram(samples[:, findex, pindex], fs=SAMPLE_RATE, window=('tukey', 0.5))
                fig = plt.figure(figsize=(16, 9), dpi=80)
                plt.xlabel("Time [sec]")
                plt.ylabel("Frequency [MHz]")
                name = "{0}_p{1}_f{2}".format(out_filename, pindex, findex)
                plt.title(name)
                plt.pcolormesh(t, f, sxx)
                plt.colorbar()
                LOG.info("Saving plot {0}".format(name))
                plt.savefig("{0}_{1}/{2}.png".format(out_filename, start_sample, name))
                fig.clear()
                plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a spectrogram across a part of an lba file")
    parser.add_argument('lba_file', type=str, help="The lba file to use")
    parser.add_argument('out_file', type=str, help="File to output spectrograms to")
    parser.add_argument('start_sample', type=int, default=0, help="The sample to start at")
    parser.add_argument('num_samples', type=int, help="The number of samples to use")
    return vars(parser.parse_args())


def main():
    args = parse_args()
    create_spectrograms(args['lba_file'], args['out_file'], args['start_sample'], args['num_samples'])


if __name__ == "__main__":
    main()
