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

import os
import sys
import argparse
import logging
import h5py
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(sys.path)

from lba import LBAFile

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)

class Preprocessor(object):

    def __init__(self):
        self.file = None
        self.outfile = None
        self.fft_window = None

    def parse_args(self):
        """
        Parse arguments to the script.
        :return: The arguments as a dict.
        """
        parser = argparse.ArgumentParser(description='Convert one or more LBA files into HDF5 files suitable for GAN training')
        parser.add_argument('file', type=str, help='Input LBA file')
        parser.add_argument('outfile', type=str, help='Output HDF5 file')
        parser.add_argument('--fft_window', type=int, help='The FFT window size to use when calculating the FFT of samples', default=2048)

        args = vars(parser.parse_args())
        self.file = args['file']
        self.outfile = args['outfile']
        self.fft_window = args['fft_window']

        if not os.path.exists(self.file):
            LOG.error('Input file does not exist: {0}'.format(self.file))
            return False

        if os.path.exists(self.outfile):
            LOG.error('Output file exists: {0}'.format(self.outfile))
            return False

        if self.fft_window <= 1:
            LOG.error('FFT size too small: {0}'.format(self.fft_window))
            return False

        LOG.info('Starting with args: {0}'.format(args))

        return True

    def output_values(self, values, label, outfile):
        try:
            dataset = outfile[label]
            dataset.resize()
        except:

    def update_min_max(self, min, max, label, outfile):
        pass

    def output_fft_batch(self, samples, ffts, outfile):

        for polarisation in range(samples.shape[2]):
            for channel in range(samples.shape[1]):

                for fft_batch_id in range(ffts):
                    fft_batch = samples[fft_batch_id * self.fft_window : (fft_batch_id + 1) * self.fft_window]

                    p_c_identifier = '{0}_{1}'.format(polarisation, channel)
                    fft_identifier = 'fft_{0}'.format(p_c_identifier)
                    abs_angle_identifier = 'abs_angle_{0}'.format(p_c_identifier)
                    norm_identifier_suffix = '{0}_norm'.format(p_c_identifier)

                    fft = np.fft.fft(fft_batch[:, channel, polarisation])

                    real = fft.real
                    min_real = np.min(real)
                    max_real = np.max(real)

                    imag = fft.imag
                    min_imag = np.min(imag)
                    max_imag = np.max(imag)

                    absolute = np.abs(fft)
                    min_absolute = np.min(absolute)
                    max_absolute = np.max(absolute)

                    angle = np.angle(fft)
                    min_angle = np.min(angle)
                    max_angle = np.max(angle)

                    self.output_values(np.concatenate((real, imag)), fft_identifier, outfile)
                    self.output_values(np.concatenate((absolute, angle)), abs_angle_identifier, outfile)
                    self.update_min_max(min_real, max_real, 'real_{0}_norm'.format(norm_identifier_suffix), outfile)
                    self.update_min_max(min_imag, max_imag, 'imag_{0}_norm'.format(norm_identifier_suffix), outfile)
                    self.update_min_max(min_absolute, max_absolute, 'abs_{0}_norm'.format(norm_identifier_suffix), outfile)
                    self.update_min_max(min_angle, max_angle, 'angle_{0}_norm'.format(norm_identifier_suffix), outfile)

    def __call__(self):
        self.parse_args()

        with open(self.file, 'r') as infile:
            lba = LBAFile(infile)

            max_samples = lba.max_samples
            max_samples -= max_samples % self.fft_window  # Ignore any samples at the end that won't fill a full fft window.

            samples_read = 0
            with h5py.File(self.outfile, 'w') as outfile:
                while samples_read < max_samples:
                    remaining_ffts = (max_samples - samples_read) // self.fft_window
                    ffts_to_read = min(remaining_ffts, 16)
                    samples_to_read = self.fft_window * ffts_to_read

                    samples = lba.read(samples_read, samples_to_read)
                    self.output_fft_batch(samples, ffts_to_read, outfile)

                    samples_read += samples_to_read



if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor()




