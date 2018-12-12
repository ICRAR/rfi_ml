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

import os
from configobj import ConfigObj
from collections import namedtuple

Default = namedtuple("Default", "default type")


class Config(object):

    defaults = {
        'USE_CUDA':                         Default(True, bool),
        'FILENAME':                         Default('../../data/At_c0p0_c0_p0_s1000000000_fft2048.hdf5', str),
        'MAX_EPOCHS':                       Default(60, int),
        'MAX_GENERATOR_AUTOENCODER_EPOCHS': Default(60, int),
        'SAMPLE_SIZE':                      Default(1024, int),  # 1024 signal samples to train on
        'TRAINING_BATCH_SIZE':              Default(4096, int),
        'TRAINING_BATCHES':                 Default(10000, int),
        'SAMPLES':                          Default(1000000000 // 2, int),  # can't load 16gb of data in you fool
        'WIDTH':                            Default(2048, int),
        'FFT':                              Default(True, bool),
        'USE_ANGLE_ABS':                    Default(False, bool),  # Convert from real, imag input to abs, angle
        'ADD_DROPOUT':                      Default(True, bool),  # if true, add dropout to the inputs before passing them into the network
        'ADD_NOISE':                        Default(False, bool)  # if true, add noise to the inputs before passing them into the network
    }

    @classmethod
    def create_default(cls, filename):
        config = ConfigObj()
        config.filename = filename

        for k, v in cls.defaults.items():
            config[k] = v.default

        config.write()

    def __init__(self, filename):
        if not os.path.exists(filename):
            self.create_default(filename)

        config = ConfigObj(filename)

        for k, v in self.defaults.items():
            try:
                value = config.get(k, v.default)
                if v.type is bool:
                    value = value.lower()
                    if value == 'false':
                        value = False
                    elif value == 'true':
                        value = True
                    else:
                        raise Exception()
                setattr(self, k, v.type(value))
            except:
                # Failed to convert config value
                raise Exception("Failed to convert config value to correct type: {0} {1}".format(k, v.type))

        if self.FFT:
            # 1000001536, next a largest multiple of width
            self.SAMPLES = self.SAMPLES - (self.SAMPLES % self.WIDTH) + self.WIDTH
            self.WIDTH *= 2  # Double width for fft (real + imag values)
            self.SAMPLES //= self.WIDTH  # samples should be split into WIDTH long chunks when using FFT
