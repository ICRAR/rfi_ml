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
import logging

LOG = logging.getLogger(__name__)


class Checkpoint(object):
    """
    Handles saving or loading of torch model state, optimiser state, and training epoch into a checkpoint file.
    Checkpoints represent the model at a particular point in time, and are used to restore the moddel state between
    training loop executions.
    """

    CHECKPOINT_PREFIX = 'checkpoint_'
    MODEL_PREFIX = 'model_save_'

    def __init__(self, filename, module_state=None, optimiser_state=None, epoch=None):
        """
        Create a checkpoint that saves to / loads from the specified checkpoint directory.
        The state parameters to this constructor are optional, but should all be provided at once.
        :param filename: The checkpoint directory to use. 'checkpoint_' is prepended to this filename.
        :param module_state: (optional) Current state of the torch model to save.
        :param optimiser_state: (optional) Current optimiser state to save.
        :param epoch: (optional) Current epoch to save.
        """
        self.directory = os.path.abspath('./{0}{1}'.format(self.CHECKPOINT_PREFIX, filename))
        self.module_state = module_state
        self.optimiser_state = optimiser_state
        self.epoch = epoch

    def load(self):
        """
        Find the latest checkpoint inside the provided checkpoint directory, and load the stored module state,
        optimiser state, and epoch into this checkpoint.
        Example Usage::

            checkpoint = Checkpoint('filename')
            if checkpoint.load():
                print('Success!')
            else:
                print('Fail!')

        :return: True if a checkpoint was loaded from the checkpoint directory, False if not
        :rtype bool
        """
        # Tries to load the latest file in given checkpoint directory

        # Get all files in the directory starting with the model prefix
        files = self._get_checkpoint_files()
        if len(files) == 0:
            return False  # Can't load, no files in there with the prefix
        latest = max(files, key=lambda f: os.path.getmtime(f))

        LOG.info("Loading: {0}".format(latest))

        data = torch.load(latest)
        self.module_state = data.get('module_state', None)
        self.optimiser_state = data.get('optimiser_state', None)
        self.epoch = data.get('epoch', None)
        return True

    def save(self):
        """
        Save the current checkpoint data to the provided checkpoint directory. The file created within the directory is
        formatted as 'model_save_{current_datetime}'. All older checkpoints in this directory are removed.
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
        """
        Gets all checkpoint files in the checkpoint directory
        :return: List containing files in the checkpoint directory that start with the model prefix.
        :rtype list
        """
        if not os.path.exists(self.directory):
            return []
        return [os.path.join(self.directory, f) for f in os.listdir(self.directory) if f.startswith(self.MODEL_PREFIX)]
