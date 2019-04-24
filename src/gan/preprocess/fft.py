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
import numpy as np
from jobs import JobQueue
from multiprocessing import Queue, Manager
from queue import Empty, Full


class HDF5FFTChannel(object):
    pass


class HDF5FFTDataSet(object):
    pass


return_queue = Queue()


class FFTJob(object):
    def __init__(self, samples, index):
        self._samples = samples
        self._index = index

    def __call__(self):
        result = np.fft.rfft(self._samples, axis=1)
        return_queue.put([self._index, result])


def parse_args()


if __name__ == "__main__":
    job_queue = JobQueue(12)

    batch_size = 4
    max_queue_checks = 100
    fft_length = 4096

    waiting_for = 0
    max_waiting_for = 10
    received = 0
    total = batch_size * 10000

    index = 0

    results = np.zeros((total, (fft_length // 2) + 1), dtype=np.complex)

    while received != total:
        while waiting_for < max_waiting_for and received + waiting_for < total:
            # Get items and place them on the queue
            try:
                data = np.random.normal(size=(batch_size, fft_length))
                print("Submitting {0}".format(index))
                job_queue.submit_no_wait(FFTJob(data, index))
                index += batch_size
                waiting_for += batch_size
            except Full:
                break  # Queue full, cannot add more

        for _ in range(max_queue_checks):
            try:
                return_index, result = return_queue.get_nowait()
                print("Got result for {0}".format(return_index))
                results[return_index:return_index + result.shape[0]] = result
                received += result.shape[0]
                waiting_for -= result.shape[0]
            except Empty:
                break  # Nothing in the queue to process

    print("Got all results")
    job_queue.join()


