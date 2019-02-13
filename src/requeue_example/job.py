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

import subprocess
import os

MAX_PROGRESS = 10
MAX_STEPS_PER_RUN = 2
PROGRESS_FILE = 'progress.txt'


def requeue():
    print('Requeue job')
    subprocess.call('enqueue.sh', shell=True, cwd=os.getcwd())


def load():
    try:
        with open(PROGRESS_FILE, 'r') as f:
            progress = f.read()
            print('Progress loaded at {0}'.format(progress))
            return int(progress)
    except IOError:
        print('No saved progress, starting at 0')
        return 0


def save_and_requeue(progress):
    with open(PROGRESS_FILE, 'w') as f:
        f.write('{0}'.format(progress))
    print('Progress saved at {0}'.format(progress))
    requeue()


def main():
    # Check in-progress run
    progress = load()

    steps_per_run = 0
    while progress < MAX_PROGRESS:
        progress += 1
        steps_per_run += 1
        print('Progress {0}'.format(progress))

        if steps_per_run >= MAX_STEPS_PER_RUN:
            save_and_requeue(progress)
            break


if __name__ == '__main__':
    main()
