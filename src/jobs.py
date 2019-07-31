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
Manages a multi-process job and worker queue.
"""

from typing import Callable
from multiprocessing import JoinableQueue, Process
import traceback
import sys


class Consumer(Process):
    def __init__(self, queue: JoinableQueue):
        """
        A consumer process gets jobs from the queue and executes them
        until it receives a None poison pill when it stops and shuts down.

        Parameters
        ----------
        queue : JoinableQueue
            The JoinableQueue that this consumer should get its jobs from
        """
        Process.__init__(self)
        self._queue = queue

    def run(self):
        """
        Run the Consumer thread, accepting jobs from the queue until a None job
        is received.
        Any uncaught exceptions raised by the job will be logged and the job
        will be cancelled.
        """
        while True:
            next_task = self._queue.get()
            if next_task is None:
                # Poison pill means shutdown this consumer
                self._queue.task_done()
                return
            # noinspection PyBroadException
            try:
                next_task()
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback)
            finally:
                self._queue.task_done()


class JobQueue(object):

    def __init__(self, num_processes: int = 8):
        """
        A job queue that contains a number of `Consumer` processes that run jobs submitted to the queue.

        Parameters
        ----------
        num_processes : int
            Number of `Consumer` processes to run for this queue.
        """
        self._queue = JoinableQueue()
        self._consumers = [Consumer(self._queue) for _ in range(num_processes)]
        for consumer in self._consumers:
            consumer.start()

    def join(self):
        """
        Join the queue, submitting a poison pill to each `Consumer` and waiting for them to all shut down.
        """
        for _ in self._consumers:
            self._queue.put(None)
        self._queue.join()

    def submit(self, job: Callable[[], None]):
        """
        Submit a job to the queue.
        This call will wait for space if the queue is full.

        Parameters
        ----------
        job : Callable[None, None]
            The job to submit to the job queue
        """
        self._queue.put(job)

    def submit_no_wait(self, job):
        """
        Submit a job to the queue without waiting.

        Parameters
        ----------
        job : Callable[None, None]
            The job to submit to the job queue

        Raises
        ----------
        JoinableQueue.Full
            If the queue is full.

        """
        self._queue.put_nowait(job)
