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
import numpy as np
from lba import LBAFile

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert an LBA file into an easier to load format")
    parser.add_argument('lba_file', type=str, help="LBA file to convert")
    parser.add_argument('outfile', type=str, help="Output file to write to")
    parser.add_argument('--all', default=False, action='store_true', help="Save all data for all channels and polarisations")
    parser.add_argument('--fft', default=False, action='store_true', help="Perform an FFT on the data and save the FFT output")
    parser.add_argument('--fft_size', type=int, default=4096, help="")
    parser.add_argument('--samples', type=int, help="Number of samples to read")
    parser.add_argument('--channel', type=int, help="Pull out a specific channel from the signal")
    parser.add_argument('--polarisation', type=int, help='Pull out a specific polarisation from the signal')
    parser.add_argument('--fft_angles_abs', default=False, action='store_true', help="If present, use the phase angles and absolute FFT values instead of the real and imaginary values")
    return vars(parser.parse_args())


def write_raw(outfile, dataset_name, samples):
    try:
        dataset = outfile[dataset_name]
        dataset.resize(dataset.shape[0] + samples.shape[0], axis=0)
        dataset[-samples.shape[0]:] = samples
    except:
        outfile.create_dataset(dataset_name, data=samples, maxshape=(None,))


def write_fft(outfile, dataset_name, samples, angles_abs):
    # Given one chunk of samples. FFT it, and write out the real and imaginary components
    fft = np.fft.fft(samples)
    out1 = fft.real
    out2 = fft.imag
    if angles_abs:
        # Use the absolute values and phase angles instead of the real and imaginary numbers
        out1 = np.abs(fft)
        out2 = np.angle(fft)
    fft = np.concatenate((out1, out2)).reshape((1, fft.shape[0] * 2))
    try:
        dataset = outfile[dataset_name]
        dataset.resize(dataset.shape[0] + fft.shape[0], axis=0)
        dataset[-fft.shape[0]:] = fft
    except:
        outfile.create_dataset(dataset_name, data=fft, maxshape=(None, fft.shape[1]))


def save_all(args):
    basename, ext = os.path.splitext(args['outfile'])
    outname = "{0}_all.hdf5".format(basename)

    try:
        os.remove(outname)
    except Exception:
        pass

    with open(args['lba_file'], 'r') as f:
        lba = LBAFile(f)
        max_samples = lba.max_samples
        CHUNK_SIZE = 1024 * 1024 * 10
        samples_read = 0

        with h5py.File(outname, 'w') as outfile:
            while samples_read < max_samples:
                to_read = min(max_samples - samples_read, CHUNK_SIZE)
                samples = lba.read(samples_read, to_read)
                samples_read += to_read

                LOG.info("{0} / {1}. {2}%".format(samples_read, max_samples, (samples_read / max_samples) * 100))

                for polarisation in range(0, samples.shape[2]):
                    polarisation_group = outfile.require_group('polarisation_{0}'.format(polarisation))
                    for channel in range(0, samples.shape[1]):
                        channel_group = polarisation_group.require_group('channel_{0}'.format(channel))

                        try:
                            dataset = channel_group['samples']
                            dataset.resize(dataset.shape[0] + samples.shape[0], axis=0)
                            dataset[-samples.shape[0]:] = samples[:, channel, polarisation]
                        except:
                            channel_group.create_dataset('samples', data=samples[:, channel, polarisation], maxshape=(None,))

def main():
    args = parse_args()
    LOG.info("Starting...")

    if args['all']:
        save_all(args)

    # Form name for outfile using base name, samples, channel, polarisation
    basename, ext = os.path.splitext(args['outfile'])
    if args['fft']:
        outname = "{0}_c{1}_p{2}_s{3}_fft{4}{5}".format(basename, args['channel'], args['polarisation'], args['samples'], args['fft_size'], ext)
    else:
        outname = "{0}_c{1}_p{2}_s{3}{4}".format(basename, args['channel'], args['polarisation'], args['samples'], ext)

    try:
        os.remove(outname)
    except Exception:
        pass

    with open(args['lba_file'], 'r') as f:
        lba = LBAFile(f)

        max_samples = args['samples']
        if args['fft']:
            CHUNK_SIZE = args['fft_size']
            max_possible_samples = lba.max_samples - (lba.max_samples % CHUNK_SIZE) # Next smallest multiple of CHUNK_SIZE
            max_samples = max_samples - (max_samples % CHUNK_SIZE) + CHUNK_SIZE  # Next largest multiple of CHUNK_SIZE
            max_samples = min(max_possible_samples, max_samples)
        else:
            CHUNK_SIZE = 1024 * 1024 * 10

        samples_read = 0
        channel = args['channel']
        polarisation = args['polarisation']

        with h5py.File(outname, 'w') as outfile:
            while samples_read < max_samples:
                to_read = min(max_samples - samples_read, CHUNK_SIZE)
                samples = lba.read(samples_read, to_read)[:, channel, polarisation]
                samples_read += to_read

                LOG.info("{0} / {1}. {2}%".format(samples_read, max_samples, (samples_read / max_samples) * 100))

                if args['fft']:
                    write_fft(outfile, 'data', samples, args['fft_angles_abs'])
                else:
                    write_raw(outfile, 'data', samples)


if __name__ == "__main__":
    main()
