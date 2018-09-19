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
Preprocess samples from an lba file and generate training data.
No fft version
Extract samples from lba files and use them directly
"""
import os
import argparse
import logging
import h5py
from lba import LBAFile

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert an LBA file into an easier to load format")
    parser.add_argument('lba_file', type=str, help="LBA file to convert")
    parser.add_argument('outfile', type=str, help="Output file to write to")
    parser.add_argument('--samples', type=int, help="Number of samples to read")
    parser.add_argument('--channel', type=int, help="Pull out a specific channel from the signal")
    parser.add_argument('--polarisation', type=int, help='Pull out a specific polarisation from the signal')
    return vars(parser.parse_args())


def main():
    args = parse_args()
    LOG.info("Starting...")

    try:
        os.remove(args['outfile'])
    except Exception:
        pass

    with open(args['lba_file'], 'r') as f:
        lba = LBAFile(f)
        CHUNK_SIZE = 1024 * 1024 * 10
        samples_read = 0
        max_samples = args['samples']

        channel = args['channel']
        polarisation = args['polarisation']

        with h5py.File(args['outfile'], 'w') as outfile:
            dataset = None

            while samples_read < max_samples:
                to_read = min(max_samples - samples_read, CHUNK_SIZE)
                samples = lba.read(samples_read, to_read)[:, channel, polarisation]
                samples_read += to_read

                LOG.info("{0} / {1}. {2}%".format(samples_read, max_samples, (samples_read / max_samples) * 100))

                if dataset is None:
                    dataset = outfile.create_dataset('data', data=samples, maxshape=(None,))
                else:
                    dataset.resize(dataset.shape[0] + samples.shape[0], axis=0)
                    dataset[-samples.shape[0]:] = samples


if __name__ == "__main__":
    main()
