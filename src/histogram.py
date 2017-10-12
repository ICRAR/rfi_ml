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
import math

import numpy as np


class Histogram(object):
    def __init__(self, data, bins=10, title=None, number_range=None, histogram_type='bars'):
        self.bins = bins
        self.title = title
        self.type = histogram_type
        self.histogram = np.histogram(np.array(data), bins=self.bins, range=number_range)

    def horizontal(self, height=4, character='|'):
        if self.title is not None:
            his = "{0}\n\n".format(self.title)
        else:
            his = ""

        if self.type == 'bars':
            bars = self.histogram[0] / float(max(self.histogram[0])) * height
            for reversed_height in reversed(range(1, height+1)):
                if reversed_height == height:
                    line = '{0} '.format(max(self.histogram[0]))
                else:
                    line = ' '*(len(str(max(self.histogram[0]))) + 1)
                for c in bars:
                    if c >= math.ceil(reversed_height):
                        line += character
                    else:
                        line += ' '
                line += '\n'
                his += line
            his += '{0:.2f}'.format(self.histogram[1][0]) + ' ' * self.bins + '{0:.2f}'.format(self.histogram[1][-1]) + '\n'
        else:
            his += ' ' * 4
            his += ''.join(['| {0:^8} '.format(n) for n in self.histogram[0]])
            his += '|\n'
            his += ' ' * 4
            his += '|----------'*len(self.histogram[0])
            his += '|\n'
            his += ''.join(['| {0:^8.2f} '.format(n) for n in self.histogram[1]])
            his += '|\n'
        return his

    def vertical(self, height=20, character='|'):
        if self.title is not None:
            his = "{0}\n\n".format(self.title)
        else:
            his = ""

        if self.type == 'bars':
            xl = ['{0:.2f}'.format(n) for n in self.histogram[1]]
            lxl = [len(l) for l in xl]
            bars = self.histogram[0] / float(max(self.histogram[0])) * height
            bars = np.rint(bars).astype('uint32')
            his += ' '*(max(bars)+2+max(lxl))+'{0}\n'.format(max(self.histogram[0]))
            for i, c in enumerate(bars):
                line = xl[i] + ' '*(max(lxl)-lxl[i])+': ' + character*c+'\n'
                his += line
        else:
            for item1, item2 in zip(self.histogram[0], self.histogram[1]):
                line = '{0:>6.2f} | {1:>5}\n'.format(item2, item1)
                his += line
        return his


if __name__ == "__main__":
    d = np.random.normal(size=1000)
    h = Histogram(d, bins=10, title='Bars Test Title')
    print h.vertical(15)
    print h.horizontal(5)

    h = Histogram(d, bins=10, title='Numbers Test Title', histogram_type='numbers', number_range=(math.floor(d.min()), math.ceil(d.max())))
    print h.vertical()
    print h.horizontal()
