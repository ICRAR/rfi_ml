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

import torch
import os
import datetime


class Checkpoint(object):
    CHECKPOINT_PREFIX = 'checkpoint_'
    MODEL_PREFIX = 'model_save_'

    def __init__(self, filename, module_state=None, optimiser_state=None, epoch=None):
        """
        Create a new checkpoint
        """
        self.directory = os.path.abspath('./{0}{1}'.format(self.CHECKPOINT_PREFIX, filename))
        self.module_state = module_state
        self.optimiser_state = optimiser_state
        self.epoch = epoch

    def load(self):
        # Tries to load the latest file in given checkpoint directory

        # Get all files in the directory starting with the model prefix
        files = self._get_checkpoint_files()
        if len(files) == 0:
            return False  # Can't load, no files in there with the prefix
        latest = max(files, key=lambda f: os.path.getmtime(f))

        data = torch.load(latest)
        self.module_state = data.get('module_state', None)
        self.optimiser_state = data.get('optimiser_state', None)
        self.epoch = data.get('epoch', None)
        return True

    def save(self):
        """
        Save the checkpoint to a file
        :param f: File descriptor or filename
        """
        save_path = os.path.join(self.directory, '{0}{1}'.format(self.MODEL_PREFIX, datetime.datetime.now()))

        os.makedirs(self.directory, exist_ok=True)

        # Remove all old checkpoints and only keep the latest
        for file in self._get_checkpoint_files():
            os.remove(file)

        torch.save({
            'module_state': self.module_state,
            'optimiser_state': self.optimiser_state,
            'epoch': self.epoch,
        }, save_path)

    def _get_checkpoint_files(self):
        if not os.path.exists(self.directory):
            return []
        return [os.path.join(self.directory, f) for f in os.listdir(self.directory) if f.startswith(self.MODEL_PREFIX)]
