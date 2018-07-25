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
import logging
import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from gan.model import Discriminator, Generator
from gan.checkpoint import Checkpoint
from gan.plots import Plots
from gan.data import generate_fake_noise, load_real_noise, generate_labels

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)

SAMPLE_SIZE = 1024  # 1024 samples to train on
TRAINING_BATCH_SIZE = 128

def train(max_epochs, use_cuda=True):

    def get_discriminator_args(data):
        return data[:, 0:SAMPLE_SIZE], data[:, SAMPLE_SIZE:]

    def get_generator_args(data):
        return data[:, 0:SAMPLE_SIZE]

    # Ensure the checkpoint directories exist
    Checkpoint.create_directory("discriminator")
    Checkpoint.create_directory("generator")

    # Create the objects we need for training
    plots = Plots(workers=2)
    # fft size is sample size * 2, one for real and one for imag
    discriminator = Discriminator(SAMPLE_SIZE, fft_size=SAMPLE_SIZE * 2)
    generator = Generator(SAMPLE_SIZE)

    if use_cuda:
        discriminator = nn.DataParallel(discriminator.cuda())
        generator = nn.DataParallel(generator.cuda())
        LOG.info("Using CUDA")
    else:
        LOG.info("Using CPU")

    criterion = nn.BCELoss()
    discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=0.0003)
    generator_optimiser = optim.Adam(generator.parameters(), lr=0.0003)
    epoch = 0

    # Try restoring previous model state if it exists
    discriminator_epoch = Checkpoint.try_restore("discriminator", discriminator, discriminator_optimiser)
    generator_epoch = Checkpoint.try_restore("generator", generator, generator_optimiser)

    if discriminator_epoch != generator_epoch:
        LOG.error("Discriminator and generator checkpoints out of sync. Ignoring")
    elif discriminator_epoch is not None:
        epoch = discriminator_epoch

    # Create two fake noise data sets, which are a normal distribution normalised between -1 and 1.
    # 10000 batches containing 1024 samples per batch
    fake_noise_data1 = DataLoader(generate_fake_noise(1000, SAMPLE_SIZE),
                                  batch_size=TRAINING_BATCH_SIZE,
                                  shuffle=True,
                                  pin_memory=use_cuda,
                                  num_workers=1)

    fake_noise_data2 = DataLoader(generate_fake_noise(1000, SAMPLE_SIZE),
                                  batch_size=TRAINING_BATCH_SIZE,
                                  shuffle=True,
                                  pin_memory=use_cuda,
                                  num_workers=1)

    real_noise_data = DataLoader(load_real_noise("../../data/v255ae_At_072_060000.lba", 1000, SAMPLE_SIZE),
                                 batch_size=TRAINING_BATCH_SIZE,
                                 shuffle=True,
                                 pin_memory=use_cuda,
                                 num_workers=1)

    real_labels = None
    fake_labels = None

    # Training loop
    while epoch < max_epochs:
        for step, (real, fake1, fake2) in enumerate(zip(real_noise_data, fake_noise_data1, fake_noise_data2)):

            step_batch_size = real.size(0)
            if real_labels is None or real_labels.size(0) != step_batch_size:
                real_labels = generate_labels(step_batch_size, [1.0, 0.0], use_cuda=use_cuda)
            if fake_labels is None or fake_labels.size(0) != step_batch_size:
                fake_labels = generate_labels(step_batch_size, [0.0, 1.0], use_cuda=use_cuda)

            # ============= Train the discriminator =============
            # Pass real noise through first - ideally the discriminator will return [1, 0]
            d_output_real = discriminator(real)
            # Pass fake noise through - ideally the discriminator will return [0, 1]
            g_output_fake1 = generator(fake1)
            d_output_fake1 = discriminator(g_output_fake1)

            # Determine the loss of the discriminator by adding up the real and fake loss and backpropagate
            d_loss_real = criterion(d_output_real, real_labels)  # How good the discriminator is on real input
            d_loss_fake = criterion(d_output_fake1, fake_labels)  # How good the discriminator is on fake input
            d_loss = d_loss_real + d_loss_fake
            discriminator.zero_grad()
            d_loss.backward()
            discriminator_optimiser.step()

            # =============== Train the generator ===============
            # Pass in fake noise to the generator and get it to generate "real" noise
            g_output_fake2 = generator(fake2)
            # Judge how good this noise is with the discriminator
            d_output_fake2 = discriminator(g_output_fake2)

            # Determine the loss of the generator using the discriminator and backpropagate
            g_loss = criterion(d_output_fake2, real_labels)
            discriminator.zero_grad()
            generator.zero_grad()
            g_loss.backward()
            generator_optimiser.step()

            if step % 10 == 0 and step > 0:
                # Report data and save checkpoint
                fmt = "Epoch [{0}/{1}], Step[{2}], d_loss_real: {3:.4f}, d_loss_fake: {4:.4f}, g_loss: {5:.4f}"
                LOG.info(fmt.format(epoch, max_epochs, step, d_loss_real, d_loss_fake, g_loss))

                plots.generate(real, g_output_fake1, g_output_fake2, epoch)

                Checkpoint.save_state("discriminator", discriminator.state_dict(), discriminator_optimiser.state_dict(), epoch)
                Checkpoint.save_state("generator", generator.state_dict(), generator_optimiser.state_dict(), epoch)
        epoch += 1


if __name__ == "__main__":
    train(10, use_cuda=False)