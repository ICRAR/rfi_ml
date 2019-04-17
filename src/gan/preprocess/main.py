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

import argparse
import logging
import os
from preprocess_lba import PreprocessReaderLBA
from hdf5_definition import HDF5Observation

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


class PreprocessorMain(object):

    def __init__(self):
        self.preprocess_readers = {
            '.lba': PreprocessReaderLBA
        }

    def _get_preprocessor(self, filename, **kwargs):
        _, ext = os.path.splitext(filename)
        constructor = self.preprocess_readers.get(ext, None)
        if constructor is None:
            raise Exception('Unknown input file type {0}'.format(ext))
        return constructor(**kwargs)

    def __call__(self, infilename, outfilename, **kwargs):
        with HDF5Observation(outfilename) as observation, open(infilename, 'r') as infile:
            try:
                preprocessor = self._get_preprocessor(infilename, **kwargs)
                preprocessor.preprocess(infilename, infile, observation)
            except:
                LOG.exception('Failed to run preprocessor')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert an input dataset into a common HDF5 format'
    )
    parser.add_argument('input', help='The input file')
    parser.add_argument('output', help='The output HDF5 file')
    parser.add_argument('--max_samples',
                        type=int,
                        default=0,
                        help='Maximum number of samples to extract from the file')
    parser.add_argument('--chunk_size',
                        type=int,
                        default=4096,
                        help='Number of samples to read from the input file at once')
    parser.add_argument('--lba_obs_file',
                        type=str,
                        help='Observation file to use for LBA files')
    parser.add_argument('--lba_sample_rate',
                        type=int,
                        help='Sample rate for LBA files. Required to parse LBA files properly')
    parser.add_argument('--lba_antenna_name',
                        type=str,
                        help="The name of the antenna def for this LBA file in the vex file. e.g. At, Mp, Pa. "
                             "Corresponds to an entry in the vex.antennas list. Used to pick the appropriate metadata "
                             "from the vex file for this LBA file")
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    LOG.info('Starting with args: {0}'.format(args))
    main = PreprocessorMain()
    main(args['input'], args['output'], **args)

