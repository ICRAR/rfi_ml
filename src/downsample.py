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
Reads an LBA file, downsamples it, and writes it out as a numpy array.
"""
import argparse
import numpy as np
import itertools
from scipy import signal
from lba import LBAFile


def parse_args():
    parser = argparse.ArgumentParser(description="Downsample an LBA file and output it as a numpy array.")
    parser.add_argument('lba_file', type=str, help="LBA file to downsample")
    parser.add_argument('output_file', type=str, help="File to output numpy array to")
    parser.add_argument('--factor', type=int, default=2, help="Downsample factor")
    parser.add_argument('--offset', type=int, default=0, help="Offset to read samples from")
    parser.add_argument('--samples', type=int, default=0, help="Number of samples to read from the file")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    with open(args['lba_file'], 'r') as f:
        lba_file = LBAFile(f)
        samples = lba_file.read(args['offset'], args['samples'])
        factor = args['factor']
        downsamples = np.zeros((samples.shape[0] // factor, samples.shape[1], samples.shape[2]))
        for pindex, f in itertools.product(range(2), range(4)):
            readsamples = samples[:, f, pindex]
            downsamples[:, f, pindex] = signal.decimate(readsamples, args['factor'])
        np.savez_compressed(args['output_file'], downsamples)


if __name__ == "__main__":
    main()