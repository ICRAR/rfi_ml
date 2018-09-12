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
Generate fake noise
Generate gaussian noise using -0.0289923828125, 1.9391296947313124 as mean and stddev. These are the
mean and stdddev of the lba files.
"""
import numpy as np


def generate_fake_noise(inputs, size):
    # TODO: Actually generate -3, -1, 1, 3 as the only pieces of data
    return np.random.normal(-0.0289923828125, 1.9391296947313124, (inputs, size)).astype(np.float32)