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
import matplotlib.pyplot as plt
import numpy as np
import h5py


def show_plot(array, title):
    # Pick a random set of data from the 2D array and show it
    index = np.random.randint(0, array.shape[1])
    sample = array[index]
    plt.title("{0} {1}".format(title, index))
    plt.plot(sample)
    plt.show()


def show_plots(filename, num_plots):
    with h5py.File(filename, 'r') as f:
        for i in range(num_plots):
            pindex = np.random.randint(0, 2)
            findex = np.random.randint(0, 4)

            show_plot(f["real"]["p{0}".format(pindex)]["f{0}".format(findex)], "Real p{0} f{1}".format(pindex, findex))
            show_plot(f["fake1"], "Fake1")
            show_plot(f["fake2"], "Fake2")


def parse_args():
    parser = argparse.ArgumentParser(description="Show plots of the data inside an hdf5 file")
    parser.add_argument('file', type=str, help="File to open")
    parser.add_argument('num_plots', type=int, help="Number of plots to show for fake and real noise")
    return vars(parser.parse_args())


def main():
    args = parse_args()
    show_plots(args['file'], args['num_plots'])


if __name__ == "__main__":
    main()