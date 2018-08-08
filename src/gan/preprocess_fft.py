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

1. FFT training data - split the data into inputs of samples and FFT each input. Save 1000 fft'd input into each
file.

2. Sample training data - split the data into chunks of samples, and save 1000 chunks per file.
"""
import argparse
import numpy as np
import h5py
from lba import LBAFile
from scipy import signal


def save_hdf5(group, d):
    for k, v in d.items():
        if isinstance(v, dict):
            save_hdf5(group.create_group(k), v)
        else:
            group.create_dataset(k, data=np.array(v))


def generate_fake_noise(total_inputs, input_size):
    # 10000, 2048
    data = np.random.normal(0, 1.0, (total_inputs, input_size)).astype(np.float32)
    # Convert to the -3 to 3 encoding we have in the lba data
    data *= 6.0
    data -= 3.0
    samples = []
    for i in range(data.shape[0]):
        fft = np.fft.fft(data[i])
        samples.append(np.concatenate((fft.real, fft.imag)).astype(np.float32))
    return np.array(samples)


def save_fft_data(filename, outfilename, sample_size, chunks_per_file):
    # Pick random position in file, where position + sample_size < max_samples
    # Read the data
    # Create a dataset for each frequency, then under each frequency, each polarisation

    with open(filename, 'r') as f:
        lba = LBAFile(f)
        with h5py.File(outfilename, 'w') as outfile:
            real = {}
            fake1 = generate_fake_noise(chunks_per_file, sample_size)
            fake2 = generate_fake_noise(chunks_per_file, sample_size)
            for _ in range(chunks_per_file):
                sample_position = np.random.randint(0, lba.max_samples - sample_size)
                samples = lba.read(sample_position, sample_size)
                for pindex in range(samples.shape[2]):
                    pdict = real.setdefault("p{0}".format(pindex), {})
                    for findex in range(samples.shape[1]):
                        flist = pdict.setdefault("f{0}".format(findex), [])
                        fft = np.fft.fft(samples[:, findex, pindex])
                        flist.append(np.concatenate((fft.real, fft.imag)))

            save_hdf5(outfile, {"fake1": fake1, "fake2": fake2, "real": real})


def parse_args():
    parser = argparse.ArgumentParser(description="Convert an LBA file into an easier to load format")
    parser.add_argument('lba_file', type=str, help="LBA file to convert")
    parser.add_argument('outfile', type=str, help="Output file to write to")
    parser.add_argument('--sample_size', type=int, help="Number of samples per chunk")
    parser.add_argument('--chunks_per_file', type=int, help="Number of chunks to read")
    return vars(parser.parse_args())


def main():
    args = parse_args()
    save_fft_data(args['lba_file'], args['outfile'], args['sample_size'], args['chunks_per_file'])


if __name__ == "__main__":
    main()