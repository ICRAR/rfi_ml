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
Load configobj configuration files to configure training.
"""

import os
from configobj import ConfigObj
from collections import namedtuple

Default = namedtuple("Default", "default type list_type")
Default.__new__.__defaults__ = (None, None, None)


class Config(object):

    defaults = {
        'USE_CUDA':                         Default(True, bool),  # True to train using the GPU, false to use the CPU
        'FILENAME':                         Default('data/processed/C148700001_fft.hdf5', str),  # HDF5 file to load data from
        'MAX_EPOCHS':                       Default(60, int),  # Max number of epochs to train the GAN for
        'MAX_AUTOENCODER_EPOCHS':           Default(60, int),  # Max number of epochs to train the autoencoder for
        'MAX_SAMPLES':                      Default(0, int),  # Maximum number of inputs to train on. Set to 0 for unlimited
        'BATCH_SIZE':                       Default(4096, int),  # Number of samples to train on per batch
        'NORMALISE':                        Default(True, bool),  # Set to true to normalise inputs
        'ADD_DROPOUT':                      Default(True, bool),  # if true, add dropout to the inputs before passing them into the network
        'ADD_NOISE':                        Default(False, bool),  # if true, add noise to the inputs before passing them into the network
        'REQUEUE_EPOCHS':                   Default(0, int),  # if > 0, perform REQUEUE_EPOCHS of training, stop, then run the REQUEUE_SCRIPT
        'REQUEUE_SCRIPT':                   Default("", str),  # if REQUEUE_EPOCHS > 0, this script will be called to requeue the training script
        'CHECKPOINT_DIRECTORY':             Default("data/checkpoints", str),
        'RESULT_DIRECTORY':                 Default("data/results/", str)
    }

    @classmethod
    def create_default(cls, filename: str):
        """
        Create a default configuration file using the provided filename.

        Parameters
        ----------
        filename : str
            The filename to save the default configuration to.

        Returns
        -------

        """
        config = ConfigObj()
        config.filename = filename

        for k, v in cls.defaults.items():
            config[k] = v.default

        config.write()

    def __init__(self, filename: str):
        """
        Handles loading of a ConfigObj configuration file, and creation of a default configuration file.
        Configuration values are parsed into attributes on this class, accessible via config.CONFIGURATION_OPTION.

        Load the configuration from the specified file.

        Parameters
        ----------
        filename : str
            The file to load the configuration from.
        """
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
