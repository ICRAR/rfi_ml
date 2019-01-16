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

from multiprocessing import JoinableQueue, Process
import traceback, sys


class Consumer(Process):
    """
    A class to process jobs from the queue
    """
    def __init__(self, queue):
        Process.__init__(self)
        self._queue = queue

    def run(self):
        """
        Sit in a loop
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

    def __init__(self, num_processes=8):
        self.queue = JoinableQueue()
        self.consumers = [Consumer(self.queue) for x in range(num_processes)]
        for consumer in self.consumers:
            consumer.start()

    def join(self):
        for consumer in self.consumers:
            self.queue.put(None)
        self.queue.join()

    def submit(self, job):
        self.queue.put(job)