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
    CHECKPOINT_PREFIX = "checkpoint_"

    @classmethod
    def get_directory(cls, model_type):
        return "{0}{1}".format(cls.CHECKPOINT_PREFIX, model_type)

    @classmethod
    def create_directory(cls, model_type):
        os.makedirs(cls.get_directory(model_type), exist_ok=True)

    @classmethod
    def try_restore(cls, checkpoint_folder, model, optimiser):
        files = [f for f in os.listdir(cls.get_directory(checkpoint_folder)) if f.startswith(cls.CHECKPOINT_PREFIX)]
        if len(files) == 0:
            return None
        return Checkpoint.load(max(files, key=lambda f: os.path.getmtime(f))).restore(model, optimiser)

    @classmethod
    def save_state(cls, model_type, model_state, optimiser_state, epoch):
        filename = os.path.join(cls.get_directory(model_type), "model_save_{0}".format(datetime.datetime.now()))
        Checkpoint(model_state, optimiser_state, epoch).save(filename)

    @staticmethod
    def load(f):
        """
        Load a checkpoint from a file
        :param f: File descriptor or filename
        :return: Loaded checkpoint
        """
        data = torch.load(f)
        return Checkpoint(data["module_state"], data["optimiser_state"], data["epoch"])

    def __init__(self, module_state=None, optimiser_state=None, epoch=None):
        """
        Create a new checkpoint
        :param module_state: Module state returned by module.state_dict()
        :param optimiser_state: Optimiser state returned by optimiser.state_dict()
        :param epoch: Training epoch
        """
        self.module_state = module_state
        self.optimiser_state = optimiser_state
        self.epoch = epoch

    def save(self, f):
        """
        Save the checkpoint to a file
        :param f: File descriptor or filename
        """
        torch.save({
            "module_state": self.module_state,
            "optimiser_state": self.optimiser_state,
            "epoch": self.epoch,
        }, f)

    def restore(self, module, optimiser):
        """
        Restore a module and optimiser from this checkpoint
        :param module: The module to restore
        :param optimiser: The optimiser to restore
        :return: The restored epoch
        """
        if self.module_state is not None:
            module.load_state_dict(self.module_state)
        if self.optimiser_state is not None:
            optimiser.load_state_dict(self.optimiser_state)
        return self.epoch