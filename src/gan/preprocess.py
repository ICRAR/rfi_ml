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
import argparse
import logging
import h5py
import numpy as np
from lba import LBAFile
from gan.noise import generate_fake_noise

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert an LBA file into an easier to load format")
    parser.add_argument('lba_file', type=str, help="LBA file to convert")
    parser.add_argument('outfile', type=str, help="Output file to write to")
    parser.add_argument('--input_size', type=int, help="Number of samples per input")
    parser.add_argument('--num_inputs', type=int, help="Total number of inputs to create")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    input_size = args['input_size']
    num_inputs = args['num_inputs']

    LOG.info("Starting...")

    with open(args['lba_file'], 'r') as f:
        lba = LBAFile(f)
        with h5py.File(args['outfile'], 'w') as outfile:
            fake1 = generate_fake_noise(input_size, num_inputs * 8)
            fake2 = generate_fake_noise(input_size, num_inputs * 8)
            real = []
            for i in range(num_inputs):
                LOG.info("Generating {0}/{1}".format(i, num_inputs))
                start = np.random.randint(0, lba.max_samples - input_size)
                samples = lba.read(start, input_size)
                for pindex in range(samples.shape[2]):
                    for findex in range(samples.shape[1]):
                        real.append(samples[:, findex, pindex])

            outfile.create_dataset("fake1", fake1)
            outfile.create_dataset("fake2", fake2)
            outfile.create_dataset("real", np.array(real))


if __name__ == "__main__":
    main()
