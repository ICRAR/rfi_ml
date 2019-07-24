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
Preprocessing pipeline step 2: Perform an FFT over the preprocessed data and save in common format.

- This process should extract multiple of FFT size chunks (e.g. fft x 10) from the input file
  and place them onto a processing queue.
- FFT calculation processes should be spawned and accept items from the queue, perform the FFT on the items,
  then place the result back on a queue destined for this process.
- This process accepts items from the destination queue (items here contain an index as to which FFT they are)
  and writes them out to the HDF5 file.
"""
import os, sys

basename = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basename, "..")))
sys.path.append(os.path.abspath(os.path.join(basename, "../..")))

import math
import argparse
import logging
import numpy as np
from hdf5_definition import HDF5Observation
from fft.hdf5_fft_definition import HDF5FFTDataSet

from jobs import JobQueue
from multiprocessing import Queue
from queue import Empty, Full

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


class FFTJob(object):
    """
    A single FFT post processing job
    """
    def __init__(self, samples, index, num_ffts, fft_size):
        self._samples = samples
        self._index = index
        self._num_ffts = num_ffts
        self._fft_size = fft_size

    def __call__(self):
        samples = self._samples.view().reshape((self._num_ffts, self._fft_size))
        result = np.fft.rfft(samples, axis=1)
        abs_result = np.abs(result)
        angle_result = np.angle(result)
        FFTPostprocessor.return_queue.put([
            self._index,
            np.stack((abs_result, angle_result), axis=-1),
            np.min(abs_result), np.max(abs_result),
            np.min(angle_result), np.max(angle_result)
        ])


class FFTPostprocessor(object):
    """
    FFT Post processor
    """

    max_queue_checks = 100
    return_queue = Queue()

    def __init__(self, workers):
        self._workers = workers
        self._fft_size = 0
        self._ffts_per_job = 0

    def _process_channel(self, job_queue, channel, out_channel, total):
        read_index = 0
        write_index = 0
        received = 0

        waiting_for = 0
        max_waiting_for = self._workers * 2 * self._ffts_per_job

        while received != total:
            while waiting_for < max_waiting_for and received + waiting_for < total:
                try:
                    # Pull the amount of data that is remaining
                    ideal_size = self._fft_size * self._ffts_per_job
                    remaining_size = channel.length - read_index
                    if ideal_size > remaining_size:
                        # Can't read back everything we want.
                        samples = channel.read_data(read_index, remaining_size)[()]
                        remainder = remaining_size % self._fft_size
                        if remainder != 0:
                            # Not a multiple, so pad up to the nearest multiple of self._fft_size
                            pad = self._fft_size - remainder
                            samples = np.append(samples, np.zeros(pad))

                        num_ffts_in_samples = samples.shape[0] // self._fft_size
                        read_size = remaining_size
                    else:
                        # Can pull out everything we want
                        samples = channel.read_data(read_index, ideal_size)[()]
                        num_ffts_in_samples = self._ffts_per_job
                        read_size = ideal_size

                    samples = samples.astype(np.float32)
                    LOG.info("Submitting: index {0}, ffts {1}".format(write_index, num_ffts_in_samples))
                    # FFTJob(samples, write_index, num_ffts_in_samples, self._fft_size)()
                    job_queue.submit_no_wait(FFTJob(samples, write_index, num_ffts_in_samples, self._fft_size))
                    read_index += read_size
                    write_index += num_ffts_in_samples
                    waiting_for += num_ffts_in_samples
                except Full:
                    break  # Queue full, cannot add more

            for _ in range(self.max_queue_checks):
                try:
                    index, result, \
                    min_abs, max_abs, \
                    min_angle, max_angle = self.return_queue.get_nowait()

                    LOG.info("Received: index {0}, ffts {1}".format(index, result.shape[0]))

                    out_channel.write_data(index, result)
                    out_channel.min_angle = min(out_channel.min_angle, min_angle)
                    out_channel.max_angle = max(out_channel.max_angle, max_angle)
                    out_channel.min_abs = min(out_channel.min_abs, min_abs)
                    out_channel.max_abs = max(out_channel.max_abs, max_abs)

                    waiting_for -= result.shape[0]
                    received += result.shape[0]
                except Empty:
                    break

    def __call__(self, infilename, outfilename, **kwargs):
        self._fft_size = kwargs.get('fft_size', 4096)
        self._ffts_per_job = kwargs.get('ffts_per_job', 128)

        job_queue = JobQueue(self._workers)

        with HDF5Observation(infilename, mode='r') as infile:
            with HDF5FFTDataSet(outfilename, mode='w') as outfile:

                num_ffts_out = infile.num_samples // self._fft_size
                if infile.num_samples % self._fft_size != 0:
                    num_ffts_out += 1

                if self._fft_size % 2 == 0:
                    # Size of output for rfft is (n/2)+1
                    fft_size_out = self._fft_size // 2 + 1
                else:
                    # Size of output for rfft is (n+1)/2
                    fft_size_out = (self._fft_size + 1) // 2

                outfile.fft_window = self._fft_size
                outfile.fft_count = num_ffts_out
                outfile.fft_input_size = fft_size_out

                for i in range(infile.num_channels):
                    channel_name = 'channel_{0}'.format(i)
                    channel = infile[channel_name]

                    out_channel = outfile.create_channel(channel_name,
                                                         shape=(num_ffts_out, fft_size_out, 2),
                                                         dtype=np.float32)
                    out_channel.min_angle = math.inf
                    out_channel.max_angle = -math.inf
                    out_channel.min_abs = math.inf
                    out_channel.max_abs = -math.inf

                    LOG.info("Processing channel {0} with {1} ffts".format(channel_name, num_ffts_out))
                    self._process_channel(job_queue, channel, out_channel, num_ffts_out)

        job_queue.join()


def parse_args():
    parser = argparse.ArgumentParser(
        description="FFT postprocessor to take a preprocessed data set and perform FFTs across each channel"
    )

    parser.add_argument("input", help="The input file to run on. This should be a file output by the preprocessor.")
    parser.add_argument("output", help="The file to write the FFT data to")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes to spawn")
    parser.add_argument("--fft_size", type=int, default=4096, help="Number of samples to include in each FFT")
    parser.add_argument("--ffts_per_job", type=int, default=128, help="Number of FFTs to perform per job submitted "
                                                                      "to worker processes")

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    LOG.info('Starting with args: {0}'.format(args))
    fft_postprocessor = FFTPostprocessor(args.get('workers', 4))
    fft_postprocessor(args['input'], args['output'], **args)
