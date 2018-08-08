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

import logging
from torch import nn, optim
from gan.model import get_models
from gan.checkpoint import Checkpoint
from gan.plots import Plots
from gan.data import generate_labels
from gan.data_fft import get_data_loaders_fft
from gan.test import TestFFT

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)

SAMPLE_SIZE = 1024  # 1024 signal samples to train on
TRAINING_BATCH_SIZE = 100
TRAINING_BATCHES = 10000


class Train(object):

    def __init__(self, use_cuda):
        # Ensure the checkpoint directories exist
        Checkpoint.create_directory("discriminator")
        Checkpoint.create_directory("generator")

        # Create the objects we need for training
        self.plots = Plots(workers=2)
        self.discriminator, self.generator = get_models(SAMPLE_SIZE)

        if use_cuda:
            self.discriminator = nn.DataParallel(self.discriminator.cuda())
            self.generator = nn.DataParallel(self.generator.cuda())
            LOG.info("Using CUDA")
        else:
            LOG.info("Using CPU")

        self.criterion = nn.BCELoss()
        self.discriminator_optimiser = optim.Adam(self.discriminator.parameters(), lr=0.0003)
        self.generator_optimiser = optim.Adam(self.generator.parameters(), lr=0.0003)
        self.epoch = 0
        self.use_cuda = use_cuda

    def print_epoch_data(self, max_epochs, step, d_loss_real, d_loss_fake, g_loss):
        # Report data and save checkpoint
        fmt = "Epoch [{0}/{1}], Step[{2}], d_loss_real: {3:.4f}, d_loss_fake: {4:.4f}, g_loss: {5:.4f}"
        LOG.info(fmt.format(self.epoch + 1, max_epochs, step, d_loss_real, d_loss_fake, g_loss))
        # self.plots.generate(real, g_output_fake1, g_output_fake2, epoch)

    def fix_labels(self, real_labels, fake_labels, real):
        step_batch_size = real.size(0)
        if real_labels is None or real_labels.size(0) != step_batch_size:
            real_labels = generate_labels(step_batch_size, [1.0, 0.0], use_cuda=self.use_cuda)
        if fake_labels is None or fake_labels.size(0) != step_batch_size:
            fake_labels = generate_labels(step_batch_size, [0.0, 1.0], use_cuda=self.use_cuda)
        return real_labels, fake_labels

    def train_fft_discriminator(self, max_epochs):
        """
        Only train the discriminator
        :param max_epochs: Max epochs to train
        :return:
        """
        real_labels = None
        fake_labels = None

        tester = TestFFT()

        LOG.info("Loading data...")
        real_noise, fake_noise1, _ = get_data_loaders_fft("train.hdf5", TRAINING_BATCH_SIZE, TRAINING_BATCHES, self.use_cuda)

        # Training loop
        LOG.info("Training start train_fft_discriminator")
        while self.epoch < max_epochs:
            for step, (real, fake) in enumerate(zip(real_noise, fake_noise1)):
                real_labels, fake_labels = self.fix_labels(real_labels, fake_labels, real)

                d_output_real = self.discriminator(real)
                d_loss_real = self.criterion(d_output_real, real_labels)  # How good the discriminator is on real input
                d_output_fake = self.discriminator(fake)
                d_loss_fake = self.criterion(d_output_fake, fake_labels)  # How good the discriminator is on fake input
                d_loss = d_loss_real + d_loss_fake
                self.discriminator.zero_grad()
                d_loss.backward()
                self.discriminator_optimiser.step()

                if step % 10 == 0:
                    self.print_epoch_data(max_epochs, step, d_loss_real, d_loss_fake, 0)

                Checkpoint.save_state("discriminator", self.discriminator.state_dict(), self.discriminator_optimiser.state_dict(), self.epoch)
            if self.epoch % 10 == 0 and self.epoch > 0:
                tester(self.discriminator)
            self.epoch += 1

    def train_fft(self, max_epochs):
        real_labels = None
        fake_labels = None

        LOG.info("Loading data...")
        real_noise, fake_noise1, fake_noise2 = get_data_loaders_fft(TRAINING_BATCH_SIZE, TRAINING_BATCHES, SAMPLE_SIZE, self.use_cuda)

        # Training loop
        LOG.info("Training start")
        while self.epoch < max_epochs:
            for step, (real, fake1, fake2) in enumerate(zip(real_noise, fake_noise1, fake_noise2)):
                real_labels, fake_labels = self.fix_labels(real_labels, fake_labels, real)

                # ============= Train the discriminator =============
                # Pass real noise through first - ideally the discriminator will return [1, 0]
                d_output_real = self.discriminator(real)
                # Pass fake noise through - ideally the discriminator will return [0, 1]
                g_output_fake1 = self.generator(fake1)
                d_output_fake1 = self.discriminator(g_output_fake1)

                # Determine the loss of the discriminator by adding up the real and fake loss and backpropagate
                d_loss_real = self.criterion(d_output_real, real_labels)  # How good the discriminator is on real input
                d_loss_fake = self.criterion(d_output_fake1, fake_labels)  # How good the discriminator is on fake input
                d_loss = d_loss_real + d_loss_fake
                self.discriminator.zero_grad()
                d_loss.backward()
                self.discriminator_optimiser.step()

                # =============== Train the generator ===============
                # Pass in fake noise to the generator and get it to generate "real" noise
                g_output_fake2 = self.generator(fake2)
                # Judge how good this noise is with the discriminator
                d_output_fake2 = self.discriminator(g_output_fake2)

                # Determine the loss of the generator using the discriminator and backpropagate
                g_loss = self.criterion(d_output_fake2, real_labels)
                self.discriminator.zero_grad()
                self.generator.zero_grad()
                g_loss.backward()
                self.generator_optimiser.step()

                if step % 5 == 0:
                    self.print_epoch_data(max_epochs, step, d_loss_real, d_loss_fake, g_loss)

            Checkpoint.save_state("discriminator", self.discriminator.state_dict(), self.discriminator_optimiser.state_dict(), self.epoch)
            Checkpoint.save_state("generator", self.generator.state_dict(), self.generator_optimiser.state_dict(), self.epoch)
            self.epoch += 1

    def __call__(self, max_epochs):
        # Try restoring previous model state if it exists
        LOG.info("Attempting checkpoint restore...")
        discriminator_epoch = Checkpoint.try_restore("discriminator", self.discriminator, self.discriminator_optimiser)
        generator_epoch = Checkpoint.try_restore("generator", self.generator, self.generator_optimiser)

        if discriminator_epoch != generator_epoch:
            LOG.error("Discriminator and generator checkpoints out of sync. Ignoring")
        elif discriminator_epoch is not None:
            self.epoch = discriminator_epoch
            LOG.info("Restored checkpoint at epoch {0}".format(self.epoch))

        self.train_fft_discriminator(max_epochs)

        # Final save after training complete
        LOG.info("Training complete, saving final model state")
        Checkpoint.create_directory("discriminator_complete")
        Checkpoint.create_directory("generator_complete")
        Checkpoint.save_state("discriminator_complete", self.discriminator.state_dict(), self.discriminator_optimiser.state_dict(), -1)
        Checkpoint.save_state("generator_complete", self.generator.state_dict(), self.generator_optimiser.state_dict(), -1)


if __name__ == "__main__":
    train = Train(use_cuda=True)
    train(max_epochs=100)