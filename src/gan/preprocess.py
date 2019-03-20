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
Ouputs the following for each file
For each combination of polarisation and channel
 - An array containing the FFTs of that channel. Each FFT is fft_window width, and do not overlap.
   the resulting array is 2x the length of the fft_window width, and contains the real and imaginary values concatenated
   in two separate blocks: [real, imaginary], [real, imaginary], ...
   All of the real values are normalised separately to the imaginary values, and are normalised to between -1 and 1
   'fft_{polarisation}_{channel}' = [fft reals 1, fft imaginaries 1, fft reals 2, fft imaginaries 2, ...]

 - An array containing the absolute value and phase angle value for each complex number in the resulting FFT. The
   resulting array is 2x the length of the fft_window width, and contains the absolute and phase angles concatenated
   in two separate blocks: [absolute, angle], [absolute, angle], ...
   All of the absolute values are normalised separately to the angle values, and are normalised to between -1 and 1
   'abs_angle_{polarisation}_{channel}' = [absolute 1, angle 1, absolute 2, angle 2, ...]

Metadata
 - file the data was loaded from
 - FFT size used
 - normalisation factors for each output combination (the min and max values)
   '{real/imag/abs/angle}_{polarisation}_{channel}_norm' = [min, max]
"""

import argparse
import logging
import os

import h5py
import numpy as np
from lba import LBAFile

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


class Preprocessor(object):
    """
    Process an LBA file into an HDF5 file, creating datasets for real + imaginary, and absolute + angle.
    Also includes the minimum and maximum values of each to allow the data to be normalised on the fly during training,
    if desired.
    """

    def __init__(self, filename, outfile, fft_window=2048, max_ffts=0, cutoff=10):
        """
        Construct a new preprocessor
        :param str filename: The LBA file to read data from
        :param str outfile: The HDF5 file to write to. If this file already exists, it will not be overwritten
        :param int fft_window: (optional) Specifies the window size, in raw samples, of the FFT to run over the raw
                    samples
        :param int max_ffts: (optional) Specifies the maximum number of FFTs to create from the lba file. Each FFT is a
                    single GAN input, so this specifies the number of GAN inputs to create. Set to 0 to create as many
                    inputs as possible from the lba file
        :param int cutoff: (optional) Specifies the number of elements at the start and end of the FFT to drop to avoid
                    artifacts.
        """
        self.file = filename
        self.outfile = outfile
        self.fft_window = fft_window
        self.max_ffts = max_ffts
        self.cutoff = cutoff
        self.ffts_output = 0

        # rfft output sizes depend on even or odd
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html
        if self.fft_window % 2 == 0:
            self.input_size = self.fft_window / 2 + 1
        else:
            self.input_size = (self.fft_window + 1) / 2

        if not os.path.exists(self.file):
            raise RuntimeError('Input file does not exist: {0}'.format(self.file))

        if os.path.exists(self.outfile):
            raise RuntimeError('Output file exists: {0}'.format(self.outfile))

        if self.fft_window <= 1:
            raise RuntimeError('FFT size too small: {0}'.format(self.fft_window))

        if self.max_ffts < 0:
            raise RuntimeError('Max FFTs < 0')

    @staticmethod
    def update_attr(value, dataset, label, function):
        """
        Updates a min/max attribute.
        :param value: The new value for the attribute.
        :param dataset: The dataset to set the attribute on.
        :param label: The attribute's label.
        :param function: Function to compare current and new attribute values.
        """
        if label not in dataset.attrs:
            dataset.attrs[label] = value
        else:
            dataset.attrs[label] = function(dataset.attrs[label], value)

    def output_fft_batch(self, samples, ffts, outfile):
        """
        Iterate over the polarisations and channels in the provided samples
        and output them to the HDF5 file.
        Also updates min and max values
        :param samples: Samples to output
        :param ffts: Number of FFTs to output for this batch.
        :param outfile: HDF5 file to output to
        """
        for polarisation in range(samples.shape[2]):
            for channel in range(samples.shape[1]):

                p_c_identifier = 'p{0}_c{1}'.format(polarisation, channel)
                if p_c_identifier not in outfile:
                    # Access is done by reading an x, y block out entirely.
                    # Chunking is not needed as the data is already stored contiguously
                    # Tested using (1, 1, self.input_size) chunks, auto chunking, and no chunking
                    # 2019-03-06 09:04:38,043:INFO:__main__:Test for file: At_1_1_size.hdf5
                    # 2019-03-06 09:05:48,477:INFO:__main__:Average iteration time: 2.3477754953333414
                    # 2019-03-06 09:05:48,479:INFO:__main__:Test for file: At_auto.hdf5
                    # 2019-03-06 09:08:37,778:INFO:__main__:Average iteration time: 5.6433112589
                    # 2019-03-06 09:08:37,779:INFO:__main__:Test for file: At_none.hdf5
                    # 2019-03-06 09:09:40,871:INFO:__main__:Average iteration time: 2.1030327905333253
                    outfile.create_dataset(p_c_identifier, shape=(self.max_ffts, 2, self.input_size - self.cutoff * 2))

                for fft_batch_id in range(ffts):
                    fft_batch = samples[fft_batch_id * self.fft_window: (fft_batch_id + 1) * self.fft_window]
                    fft = np.fft.rfft(fft_batch[:, channel, polarisation])[self.cutoff:-self.cutoff]

                    absolute = np.abs(fft)
                    angle = np.angle(fft)

                    outfile[p_c_identifier][self.ffts_output + fft_batch_id] = np.stack((absolute, angle))

                    # Store a minimum for all items in a particular channel and polarisation,
                    # and also store a minimum for all items

                    minimum = np.min(absolute)
                    self.update_attr(minimum, outfile[p_c_identifier], 'min_abs', min)
                    self.update_attr(minimum, outfile, 'min_abs', min)
                    maximum = np.max(absolute)
                    self.update_attr(maximum, outfile[p_c_identifier], 'max_abs', max)
                    self.update_attr(maximum, outfile, 'max_abs', max)

                    minimum = np.min(angle)
                    self.update_attr(minimum, outfile[p_c_identifier], 'min_angle', min)
                    self.update_attr(minimum, outfile, 'min_angle', min)
                    maximum = np.max(angle)
                    self.update_attr(maximum, outfile[p_c_identifier], 'max_angle', max)
                    self.update_attr(maximum, outfile, 'max_angle', max)

        self.ffts_output += ffts

    def __call__(self):
        """
        Run the preprocessor
        """
        with open(self.file, 'r') as infile:
            lba = LBAFile(infile)

            max_samples = lba.max_samples
            # Ignore any samples at the end that won't fill a full fft window.
            max_samples -= max_samples % self.fft_window

            if self.max_ffts > 0:
                max_samples = min(self.fft_window * self.max_ffts, max_samples)

            max_ffts = max_samples // self.fft_window
            if self.max_ffts == 0:
                self.max_ffts = max_ffts  # Get the max FFTs from the lba file as the user has not specified

            samples_read = 0
            with h5py.File(self.outfile, 'w') as outfile:
                outfile.attrs['fft_window'] = self.fft_window
                outfile.attrs['samples'] = max_samples
                outfile.attrs['fft_count'] = max_ffts
                outfile.attrs['input_size'] = self.input_size
                outfile.attrs['cutoff'] = self.cutoff
                while samples_read < max_samples:
                    remaining_ffts = (max_samples - samples_read) // self.fft_window
                    LOG.info("Processed {0} out of {1} fft windows".format(max_ffts - remaining_ffts, max_ffts))

                    ffts_to_read = min(remaining_ffts, 128)
                    samples_to_read = self.fft_window * ffts_to_read

                    samples = lba.read(samples_read, samples_to_read)
                    self.output_fft_batch(samples, ffts_to_read, outfile)

                    samples_read += samples_to_read

                LOG.info("Processed {0} out of {0} fft windows".format(max_ffts, max_ffts))


def parse_args():
    """
    Parse arguments to the script.
    :return: The arguments as a dict.
    :rtype dict
    """
    parser = argparse.ArgumentParser(
        description='Convert one or more LBA files into HDF5 files suitable for GAN training')
    parser.add_argument('file', type=str, help='Input LBA file')
    parser.add_argument('outfile', type=str, help='Output HDF5 file')
    parser.add_argument('--fft_window',
                        type=int,
                        help='The FFT window size to use when calculating the FFT of samples',
                        default=2048)
    parser.add_argument('--max_ffts',
                        type=int,
                        help='Max number of FFTs create. 0 is use all available data',
                        default=0)
    parser.add_argument('--fft_cutoff',
                        type=int,
                        help='Number of elements at the start and the end of the FFT to drop to avoid artifacts',
                        default=0)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    LOG.info('Starting with args: {0}'.format(args))
    try:
        preprocessor = Preprocessor(args['file'], args['outfile'], args['fft_window'], args['max_ffts'], args['fft_cutoff'])
        preprocessor()
    except RuntimeError as e:
        LOG.exception("Failed to run preprocessor", e)
