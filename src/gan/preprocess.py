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
        self.max_ffts = None

    def parse_args(self):
        """
        Parse arguments to the script.
        :return: The arguments as a dict.
        """
        parser = argparse.ArgumentParser(description='Convert one or more LBA files into HDF5 files suitable for GAN training')
        parser.add_argument('file', type=str, help='Input LBA file')
        parser.add_argument('outfile', type=str, help='Output HDF5 file')
        parser.add_argument('--fft_window', type=int, help='The FFT window size to use when calculating the FFT of samples', default=2048)
        parser.add_argument('--max_ffts', type=int, help='Max number of FFTs create. 0 is use all available data', default=0)

        args = vars(parser.parse_args())
        self.file = args['file']
        self.outfile = args['outfile']
        self.fft_window = args['fft_window']
        self.max_ffts = args['max_ffts']

        if not os.path.exists(self.file):
            LOG.error('Input file does not exist: {0}'.format(self.file))
            return False

        if os.path.exists(self.outfile):
            LOG.error('Output file exists: {0}'.format(self.outfile))
            return False

        if self.fft_window <= 1:
            LOG.error('FFT size too small: {0}'.format(self.fft_window))
            return False

        if self.max_ffts < 0:
            LOG.error('Max FFTs < 0')
            return False

        LOG.info('Starting with args: {0}'.format(args))

        return True

    @staticmethod
    def output_values(values, label, outfile):
        """
        Outputs a new set of fft data, either real + imag or absolute + angle.
        The two input parts are concatenated together into the 'values' parameter
        :param values: Values to add to the dataset
        :param label: Dataset name
        :param outfile: HDF5 file
        """
        try:
            dataset = outfile[label]
            dataset.resize(dataset.shape[0] + values.shape[0], axis=0)
            dataset[-values.shape[0]:] = values.astype(np.float32)
        except:
            outfile.create_dataset(label, data=values.astype(np.float32), maxshape=(None,), chunks=True)

    @staticmethod
    def update_attr(value, dataset, label, function):
        try:
            if not label in dataset.attrs:
                dataset.attrs[label] = value
            else:
                dataset.attrs[label] = function(dataset.attrs[label], value)
        except Exception as e:
            LOG.error("Can't update attr {0}".format(label))

    def output_fft_batch(self, samples, ffts, outfile):
        for polarisation in range(samples.shape[2]):
            for channel in range(samples.shape[1]):

                for fft_batch_id in range(ffts):
                    fft_batch = samples[fft_batch_id * self.fft_window : (fft_batch_id + 1) * self.fft_window]

                    fft = np.fft.fft(fft_batch[:, channel, polarisation])

                    real = fft.real
                    real = real[0:real.shape[0] // 2]
                    imag = fft.imag
                    absolute = np.abs(fft)
                    absolute = absolute[0:absolute.shape[0] // 2]
                    angle = np.angle(fft)

                    p_c_identifier = 'p{0}_c{1}'.format(polarisation, channel)
                    fft_label = '{0}_real_imag'.format(p_c_identifier)
                    abs_angle_label = '{0}_abs_angle'.format(p_c_identifier)
                    self.output_values(np.concatenate((real, imag)), fft_label, outfile)
                    self.output_values(np.concatenate((absolute, angle)), abs_angle_label, outfile)

                    # Store a minimum for all items in a particular channel and polarisation,
                    # and also store a minimum for all items
                    minimum = np.min(real)
                    self.update_attr(minimum, outfile[fft_label], 'min_real', min)
                    self.update_attr(minimum, outfile, 'min_real', min)
                    maximum = np.max(real)
                    self.update_attr(maximum, outfile[fft_label], 'max_real', max)
                    self.update_attr(maximum, outfile, 'max_real', max)

                    minimum = np.min(imag)
                    self.update_attr(minimum, outfile[fft_label], 'min_imag', min)
                    self.update_attr(minimum, outfile, 'min_imag', min)
                    maximum = np.max(imag)
                    self.update_attr(maximum, outfile[fft_label], 'max_imag', max)
                    self.update_attr(maximum, outfile, 'max_imag', max)

                    minimum = np.min(absolute)
                    self.update_attr(minimum, outfile[abs_angle_label], 'min_abs', min)
                    self.update_attr(minimum, outfile, 'min_abs', min)
                    maximum = np.max(absolute)
                    self.update_attr(maximum, outfile[abs_angle_label], 'max_abs', max)
                    self.update_attr(maximum, outfile, 'max_abs', max)

                    minimum = np.min(angle)
                    self.update_attr(minimum, outfile[abs_angle_label], 'min_angle', min)
                    self.update_attr(minimum, outfile, 'min_angle', min)
                    maximum = np.max(angle)
                    self.update_attr(maximum, outfile[abs_angle_label], 'max_angle', max)
                    self.update_attr(maximum, outfile, 'max_angle', max)

    def __call__(self):
        if not self.parse_args():
            return

        with open(self.file, 'r') as infile:
            lba = LBAFile(infile)

            max_samples = lba.max_samples
            max_samples -= max_samples % self.fft_window  # Ignore any samples at the end that won't fill a full fft window.

            if self.max_ffts > 0:
                max_samples = min(self.fft_window * self.max_ffts, max_samples)

            max_ffts = max_samples // self.fft_window

            samples_read = 0
            with h5py.File(self.outfile, 'w') as outfile:
                outfile.attrs['fft_window'] = self.fft_window
                outfile.attrs['samples'] = max_samples
                outfile.attrs['fft_count'] = max_ffts
                outfile.attrs['size_first'] = self.fft_window // 2
                outfile.attrs['size_second'] = self.fft_window
                outfile.attrs['size'] = self.fft_window + self.fft_window // 2
                while samples_read < max_samples:
                    remaining_ffts = (max_samples - samples_read) // self.fft_window
                    LOG.info("Processed {0} out of {1} fft windows".format(max_ffts - remaining_ffts, max_ffts))

                    ffts_to_read = min(remaining_ffts, 128)
                    samples_to_read = self.fft_window * ffts_to_read

                    samples = lba.read(samples_read, samples_to_read)
                    self.output_fft_batch(samples, ffts_to_read, outfile)

                    samples_read += samples_to_read

                LOG.info("Processed {0} out of {0} fft windows".format(max_ffts, max_ffts))


if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor()




