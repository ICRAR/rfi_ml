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
from gan.model import Discriminator, Generator
from gan.checkpoint import Checkpoint
from gan.plots import Plots

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)

SAMPLE_SIZE = 1024  # 1024 samples to train on


def train(max_epochs):
    # Ensure the checkpoint directories exist
    Checkpoint.create_directory("discriminator")
    Checkpoint.create_directory("generator")

    # Create the objects we need for training
    plots = Plots(2)
    discriminator = Discriminator(1024, fft_size)
    generator = Generator(1024)
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

    # Training loop
    while epoch < max_epochs:
        for step, (noise, fake1, fake2) in enumerate(zip(noise_data, fake_noise_data1, fake_noise_data2)):

            # ============= Train the discriminator =============
            d_output_real = discriminator(noise)
            d_loss_real = criterion(d_output_real, real_labels)

            g_output_fake1 = generator(fake1)
            d_output_fake1 = discriminator(g_output_fake1)
            d_loss_fake = criterion(d_output_fake1, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            discriminator.zero_grad()
            d_loss.backward()
            discriminator_optimiser.step()

            # =============== Train the generator ===============
            g_output_fake2 = generator(fake2)
            d_output_fake2 = discriminator(g_output_fake2)
            g_loss = criterion(d_output_fake2, real_labels)

            discriminator.zero_grad()
            generator.zero_grad()
            g_loss.backward()
            generator_optimiser.step()

            if step % 10 == 0 and step > 0:
                # Report data and save checkpoint
                fmt = "Epoch [{0}/{1}], Step[{2}], d_loss_real: {3:.4f}, d_loss_fake: {4:.4f}, g_loss: {5:.4f}"
                LOG.info(fmt.format(epoch, max_epochs, step, d_loss_real, d_loss_fake, g_loss))

                plots.generate(noise, g_output_fake1, g_output_fake2, epoch)

                Checkpoint.save_state("discriminator", discriminator.state_dict(), discriminator_optimiser.state_dict(), epoch)
                Checkpoint.save_state("generator", generator.state_dict(), generator_optimiser.state_dict(), epoch)
        epoch += 1
