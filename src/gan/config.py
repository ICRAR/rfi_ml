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

Default = namedtuple("Default", "default type list_type")
Default.__new__.__defaults__ = (None, None, None)


class Config(object):

    defaults = {
        'USE_CUDA':                         Default(True, bool),  # True to train using the GPU, false to use the CPU
        'FILENAME':                         Default('../../data/At_c0p0_c0_p0_s1000000000_fft2048.hdf5', str),  # HDF5 file to load data from
        'DATA_TYPE':                        Default('real_imag', str),  # Type of data to read from the HDF5 files. Either 'real_imag' or 'abs_angle'
        'MAX_EPOCHS':                       Default(60, int),  # Max number of epochs to train the GAN for
        'MAX_GENERATOR_AUTOENCODER_EPOCHS': Default(60, int),  # Max number of epochs to train the generator autoencoder for
        'MAX_SAMPLES':                      Default(0, int),  # Maximum number of inputs to train on. Set to 0 for unlimited
        'BATCH_SIZE':                       Default(4096, int),  # Number of samples to train on per batch
        'POLARISATIONS':                    Default([0, 1], list, int),  # Which polarisations should be used?
        'FREQUENCIES':                      Default([0, 1, 2, 3], list, int),  # Which frequencies should be used?
        'FULL_FIRST':                       Default(False, bool),  # Set to true use the full set of real / absolute values. False to only use one half.
        'NORMALISE':                        Default(True, bool),  # Set to true to normalise inputs
        'ADD_DROPOUT':                      Default(True, bool),  # if true, add dropout to the inputs before passing them into the network
        'ADD_NOISE':                        Default(False, bool),  # if true, add noise to the inputs before passing them into the network
        'REQUEUE_EPOCHS':                   Default(0, int),  # if > 0, perform REQUEUE_EPOCHS of training, stop, then run the REQUEUE_SCRIPT
        'REQUEUE_SCRIPT':                   Default("", str)  # if REQUEUE_EPOCHS > 0, this script will be called to requeue the training script
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
                elif v.type is list:
                    # Convert all to the list type
                    value = map(v.list_type, value)
                setattr(self, k, v.type(value))
            except:
                # Failed to convert config value
                raise Exception("Failed to convert config value to correct type: {0} {1}".format(k, v.type))


if __name__ == '__main__':
    Config.create_default('gan_config.settings')