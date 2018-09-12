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

import sys
import os
base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

import numpy as np
import torch
import logging
from torch.utils.data import DataLoader, Dataset
from lba import LBAFile
from gan import USE_CUDA

LOG = logging.getLogger(__name__)


class NoiseDataset(Dataset):
    def __init__(self, samples, width):
        self.samples = samples
        self.width = width

    def __getitem__(self, item):
        return Data.generate_fake_noise(self.samples, self.width)

    def __len__(self):
        return self.samples  # Doesn't matter as we simply re-generate the noise each time


# class LBADataset(Dataset):
#     """
#     Opens an LBA file and will load data from that file on demand.
#     This isn't fast, but it should decrease the total amount of memory
#     required by the application
#     """
#
#     def __init__(self, filename, width):
#         self.width = width
#         self.file = open(filename, 'r')
#         self.lba = LBAFile(self.file)
#
#     def __del__(self):
#         self.file.close()
#         del self.lba
#
#     def __getitem__(self, item):
#         return self.lba.read(item * self.width, self.width)
#
#     def __len__(self):
#         return self.lba.max_samples // self.width


class Data(object):

    def __init__(self, filename, samples, width, batch_size):
        self.width = width
        self.samples = samples
        self.noise = None  # Fake generated gaussian noise
        self.data = None  # Data read in from LBA / HDF5 file

        self.noise = DataLoader(NoiseDataset(self.samples, self.width),
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=USE_CUDA,
                                num_workers=1)

        # fake_noise_data2 = DataLoader(self.generate_fake_noise(self.samples, self.width),
        #                               batch_size=batch_size,
        #                               shuffle=True,
        #                               pin_memory=self.USE_CUDA,
        #                               num_workers=1)

        LOG.info("Loading real noise data...")
        self.data = DataLoader(self.load_data(filename, self.samples, self.width),
                               batch_size=batch_size,
                               shuffle=True,
                               pin_memory=USE_CUDA,
                               num_workers=1)

    def __iter__(self):
        return zip(self.noise, self.data)

    @staticmethod
    def normalise(array):
        array -= np.min(array)
        array /= np.max(array)
        array *= 2.0
        array -= 1.0
        return array

    @staticmethod
    def generate_fake_noise(samples, width):
        """
        Generate a bunch of gaussian noise for training
        :return: A torch data set of gaussian noise
        """
        data = np.random.normal(0, 1.0, (samples, width)).astype(np.float32)
        return Data.normalise(data)

    @classmethod
    def load_data(cls, filename, num_samples, width, frequency=None, polarisation=None):
        """
        Load noise data from the specified file
        :param filename: Filename to load data from
        :param num_samples: Number of training data inputs to load from the file
        :param width: Number of elements per training data input
        :return:
        """
        data = np.zeros((num_samples, width), dtype=np.float32)
        with open(filename, 'r') as f:
            lba = LBAFile(f)
            # Get a bunch of random indexes into the file that will not overflow if we read batch_size samples from
            # that index onward
            indexes = (np.random.rand(num_samples) * (lba.max_samples - width)).astype(int)

            # For each batch, either use the provided frequency and polarisation values,
            # or pick randomly from the 4 frequencies and 2 polarisations.
            frequency_indexes = (np.random.rand(num_samples) * 3).astype(int) \
                if frequency is None else np.repeat(frequency, num_samples)

            polarisation_indexes = (np.random.rand(num_samples) * 2).astype(int) \
                if polarisation is None else np.repeat(polarisation, num_samples)

            # Get data for each batch
            for batch in range(num_samples):
                if batch % 100 == 0 or batch == num_samples - 1:
                    LOG.info("Loading real data batch {0} / {1}".format(batch + 1, num_samples))
                lba_data = lba.read(indexes[batch], width)
                data[batch] = lba_data[:, frequency_indexes[batch], polarisation_indexes[batch]]

        return cls.normalise(data)

    def generate_labels(self, num_samples, pattern):
        var = torch.FloatTensor([pattern] * num_samples)
        return var.cuda() if USE_CUDA else var


if __name__ == "__main__":
    data_loader = Data("../../data/v255ae_At_072_060000.lba", 1000, 2048, 64)
    count = 0
    for data, noise in data_loader:
        count += 1
        print(data, noise)
    print(count)
