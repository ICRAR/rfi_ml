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
Preprocessing step 1: Converting a raw input file into the `src.preprocess.hdf5_definition.HDF5Observation` format.

Run this file directly to invoke the preprocessor.
"""

import os
import argparse
import logging
import tarfile

from preprocess.preprocess_india_tar import PreprocessReaderIndiaTAR
from .preprocess_lba import PreprocessReaderLBA
from .preprocess_india_txt import PreprocessReaderIndiaTXT
from .hdf5_definition import HDF5Observation

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)

_preprocess_readers = {
    '.lba': PreprocessReaderLBA,
    '.txt': PreprocessReaderIndiaTXT,
    'india_tar': PreprocessReaderIndiaTAR
}


def _get_preprocessor(filename, **kwargs):
    ext = kwargs.get("mode", None)
    if ext is None:
        _, ext = os.path.splitext(filename)
    constructor = _preprocess_readers.get(ext, None)
    if constructor is None:
        raise Exception('Unknown input file type {0}'.format(ext))
    return constructor(**kwargs), ext[1:]


def main(infilename, outfilename, **kwargs):
    """
    Run the preprocessor on the specified file.
    The preprocessor is chosen based on the extension of the file to be loaded.

    Parameters
    ----------
    infilename : str
        The input file to run on.

    outfilename : str
        The output filename to write to
    kwargs
        Arguments for the chosen preprocessor as provided by `parse_args`
    """
    if kwargs.get("mode") == "india_tar":
        # India tar produces multiple output files from a single input file
        with tarfile.open(infilename, 'r') as f:
            for info in map(lambda x: x.name.startswith("C") and x.name.endswith("txt"), f.getmembers()):
                tempfilename = f"{os.tmpnam()}.txt"
                trueoutfilename = os.path.join(outfilename, os.path.basename(info.name))
                print(f"Extracting {info.name} to {tempfilename}, then producing {trueoutfilename} as output")
                with open(tempfilename, 'w') as tmp:
                    f.extractfile(info).readinto(tmp)
                args_copy = {**kwargs, "mode": ".txt"}
                main(tempfilename, trueoutfilename, args_copy)
    else:
        with HDF5Observation(outfilename, mode='w') as observation, open(infilename, 'r') as infile:
            try:
                preprocessor, file_type = _get_preprocessor(infilename, **kwargs)

                observation.write_defaults()
                observation.original_file_name = os.path.basename(infilename)
                observation.original_file_type = file_type

                preprocessor.preprocess(infilename, infile, observation)
            except:
                LOG.exception('Failed to run preprocessor')


def parse_args():
    """
    Parse command line arguments for the program

    General arguments:
    ```text
    input: The input file to load.

    output: The file to output to.

    --max_samples: Maximum number of samples to extract from the input file.

    --chunk_size: Number of samples to read from the input file at once.
                  Default 4096.

    --sample_rate: Sample rate in Hz, needed for the LBA and India TXT parsers.
    ```

    LBA Only:
    ```text
    --lba_obs_file: Observation file to use for LBA files.

    --lba_antenna_name: The name of the antenna def for this LBA file in the
                        vex file. e.g. At, Mp, Pa. Corresponds to an entry in
                        the vex.antennas list. Used to pick the appropriate
                        metadata from the vex file for this LBA file
    ```
    """
    parser = argparse.ArgumentParser(
        description='Convert an input dataset into a common HDF5 format'
    )
    parser.add_argument('input', help='The input file')
    parser.add_argument('output', help='The output HDF5 file')
    parser.add_argument('--mode',
                        type=str,
                        help='Force a specific parsing mode')
    parser.add_argument('--max_samples',
                        type=int,
                        default=0,
                        help='Maximum number of samples to extract from the input file')
    parser.add_argument('--chunk_size',
                        type=int,
                        default=4096,
                        help='Number of samples to read from the input file at once')

    parser.add_argument('--sample_rate',
                        type=int,
                        help='Sample rate in Hz, needed for the LBA and India TXT parsers')

    # LBA arguments
    parser.add_argument('--lba_obs_file',
                        type=str,
                        help='Observation file to use for LBA files')
    parser.add_argument('--lba_antenna_name',
                        type=str,
                        help="The name of the antenna def for this LBA file in the vex file. e.g. At, Mp, Pa. "
                             "Corresponds to an entry in the vex.antennas list. Used to pick the appropriate metadata "
                             "from the vex file for this LBA file")

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    LOG.info('Starting with args: {0}'.format(args))
    main(args['input'], args['output'], **args)

