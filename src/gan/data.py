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

import numpy as np
import scipy.fftpack as fft
import torch
import logging
from torch.utils.data import DataLoader
from lba import LBAFile

LOG = logging.getLogger(__name__)


def normalise(array):
    array -= np.min(array)
    array /= np.max(array)
    array *= 2.0
    array -= 1.0
    return array


def fft_samples(samples):
    f = fft.fft(samples, axis=1)
    return np.concatenate((samples, f.real, f.imag), axis=1)


def generate_fake_noise(num_batches, batch_size):
    """
    Generate a bunch of gaussian noise for training
    :param num_samples: Number of samples to generate
    :return: A torch data set of gaussian noise
    """
    data = np.random.normal(0, 1.0, (num_batches, batch_size)).astype(np.float32)
    data = normalise(data)
    return data


def generate_labels(batch_size, pattern, use_cuda):
    new_list = [pattern] * batch_size
    var = torch.FloatTensor(new_list)
    return var.cuda() if use_cuda else var


def load_real_noise(filename, num_batches, batch_size, frequency=None, polarisation=None):
    """
    Load noise data from the specified file
    :param filename:
    :param num_samples:
    :param batch_size:
    :param use_cuda:
    :return:
    """
    data = np.zeros((num_batches, batch_size), dtype=np.float32)
    with open(filename, 'r') as f:
        lba = LBAFile(f)
        # Get a bunch of random indexes into the file that will not overflow if we read batch_size samples from
        # that index onward
        indexes = (np.random.rand(num_batches) * (lba.max_samples - batch_size)).astype(int)

        # For each batch, either use the provided frequency and polarisation values,
        # or pick randomly from the 4 frequencies and 2 polarisations.
        frequency_indexes = (np.random.rand(num_batches) * 3).astype(int) \
            if frequency is None else np.repeat(frequency, num_batches)

        polarisation_indexes = (np.random.rand(num_batches) * 2).astype(int) \
            if polarisation is None else np.repeat(polarisation, num_batches)

        # Get data for each batch
        for batch in range(num_batches):
            if batch % 100 == 0 or batch == num_batches - 1:
                LOG.info("Loading real data batch {0} / {1}".format(batch + 1, num_batches))
            lba_data = lba.read(indexes[batch], batch_size)
            data[batch] = lba_data[:, frequency_indexes[batch], polarisation_indexes[batch]]

    data = normalise(data)

    return data


def get_data_loaders(num_batches, training_batch_size, sample_size, use_cuda):
    # Create two fake noise data sets, which are a normal distribution normalised between -1 and 1.
    LOG.info("Generating fake noise data...")
    fake_noise_data1 = DataLoader(generate_fake_noise(training_batch_size * num_batches, sample_size),
                                  batch_size=training_batch_size,
                                  shuffle=True,
                                  pin_memory=use_cuda,
                                  num_workers=1)

    fake_noise_data2 = DataLoader(generate_fake_noise(training_batch_size * num_batches, sample_size),
                                  batch_size=training_batch_size,
                                  shuffle=True,
                                  pin_memory=use_cuda,
                                  num_workers=1)

    LOG.info("Loading real noise data...")
    real_noise_data = DataLoader(load_real_noise("../data/v255ae_At_072_060000.lba", training_batch_size * num_batches, sample_size),
                                 batch_size=training_batch_size,
                                 shuffle=True,
                                 pin_memory=use_cuda,
                                 num_workers=1)

    return real_noise_data, fake_noise_data1, fake_noise_data2


if __name__ == "__main__":
    SAMPLE_SIZE = 16
    loader = DataLoader(generate_fake_noise(1000, SAMPLE_SIZE), batch_size=128)
    fake = next(iter(loader))

    samples = fake[0, 0:SAMPLE_SIZE]
    reals = fake[0, SAMPLE_SIZE:SAMPLE_SIZE * 2]
    imags = fake[0, SAMPLE_SIZE * 2:]

    loader = DataLoader(load_real_noise("../../data/v255ae_At_072_060000.lba", 1000, SAMPLE_SIZE), batch_size=128)
    real = next(iter(loader))

    samples = real[0, 0:SAMPLE_SIZE]
    reals = real[0, SAMPLE_SIZE:SAMPLE_SIZE * 2]
    imags = real[0, SAMPLE_SIZE * 2:]

    pass