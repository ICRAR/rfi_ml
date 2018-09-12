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
from gan.checkpoint import Checkpoint
from gan.data import Data
from gan.models.single_polarisation_single_frequency import Generator, Discriminator

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)

SAMPLE_SIZE = 1024  # 1024 signal samples to train on
TRAINING_BATCH_SIZE = 100
TRAINING_BATCHES = 10000


class Train(object):
    USE_CUDA = True

    def __init__(self, samples, width, batch_size):
        # Ensure the checkpoint directories exist
        Checkpoint.create_directory("discriminator")
        Checkpoint.create_directory("generator")

        LOG.info("Creating models...")
        self.discriminator = Discriminator(width)
        self.generator = Generator(width)

        if self.USE_CUDA:
            self.discriminator = nn.DataParallel(self.discriminator.cuda())
            self.generator = nn.DataParallel(self.generator.cuda())
            LOG.info("Using CUDA")
        else:
            LOG.info("Using CPU")

        self.discriminator_optimiser = optim.Adam(self.discriminator.parameters(), lr=0.0003)
        self.generator_optimiser = optim.Adam(self.generator.parameters(), lr=0.0003)

        self.samples = samples
        self.width = width
        self.batch_size = batch_size

    def train(self, filename, max_epochs):
        # Try restoring previous model state if it exists
        LOG.info("Attempting checkpoint restore...")
        start_epoch = 0
        try:
            discriminator_epoch = Checkpoint.try_restore("discriminator", self.discriminator, self.discriminator_optimiser)
            generator_epoch = Checkpoint.try_restore("generator", self.generator, self.generator_optimiser)
            if discriminator_epoch != generator_epoch:
                LOG.error("Discriminator and generator checkpoints out of sync. Ignoring")
            elif discriminator_epoch is not None:
                start_epoch = discriminator_epoch
            LOG.info("Restored checkpoint at epoch {0}".format(start_epoch))
        except Exception as e:
            LOG.error("Failed to restore discriminator and generator {0}".format(e))

        self._train(filename, start_epoch, max_epochs)

        # Final save after training complete
        LOG.info("Training complete, saving final model state")
        Checkpoint.create_directory("discriminator_complete")
        Checkpoint.create_directory("generator_complete")
        Checkpoint.save_state("discriminator_complete", self.discriminator.state_dict(), self.discriminator_optimiser.state_dict(), -1)
        Checkpoint.save_state("generator_complete", self.generator.state_dict(), self.generator_optimiser.state_dict(), -1)

    def _train(self, filename, start_epoch, max_epochs):
        real_labels = None
        fake_labels = None

        criterion = nn.BCELoss()
        epoch = start_epoch

        LOG.info("Loading data...")
        data = Data(filename, self.samples, self.width, self.batch_size)

        # Training loop
        LOG.info("Training start")
        while epoch < max_epochs:
            for step, (data, noise) in enumerate(data):
                batch_size = data.size(0)
                if real_labels is None or real_labels.size(0) != batch_size:
                    real_labels = data.generate_labels(batch_size, [1.0, 0.0], use_cuda=self.USE_CUDA)
                if fake_labels is None or fake_labels.size(0) != batch_size:
                    fake_labels = data.generate_labels(batch_size, [0.0, 1.0], use_cuda=self.USE_CUDA)

                # ============= Train the discriminator =============
                # Pass real noise through first - ideally the discriminator will return [1, 0]
                d_output_real = self.discriminator(data)
                # Pass fake noise through - ideally the discriminator will return [0, 1]
                g_output_fake1 = self.generator(noise)
                d_output_fake1 = self.discriminator(g_output_fake1)

                # Determine the loss of the discriminator by adding up the real and fake loss and backpropagate
                d_loss_real = criterion(d_output_real, real_labels)  # How good the discriminator is on real input
                d_loss_fake = criterion(d_output_fake1, fake_labels)  # How good the discriminator is on fake input
                d_loss = d_loss_real + d_loss_fake
                self.discriminator.zero_grad()
                d_loss.backward()
                self.discriminator_optimiser.step()

                # =============== Train the generator ===============
                # Pass in fake noise to the generator and get it to generate "real" noise
                g_output_fake2 = self.generator(noise)
                # Judge how good this noise is with the discriminator
                d_output_fake2 = self.discriminator(g_output_fake2)

                # Determine the loss of the generator using the discriminator and backpropagate
                g_loss = criterion(d_output_fake2, real_labels)
                self.discriminator.zero_grad()
                self.generator.zero_grad()
                g_loss.backward()
                self.generator_optimiser.step()

                if step % 5 == 0:
                    # Report data and save checkpoint
                    fmt = "Epoch [{0}/{1}], Step[{2}], d_loss_real: {3:.4f}, d_loss_fake: {4:.4f}, g_loss: {5:.4f}"
                    LOG.info(fmt.format(epoch + 1, max_epochs, step, d_loss_real, d_loss_fake, g_loss))
                    # self.plots.generate(real, g_output_fake1, g_output_fake2, epoch)

            Checkpoint.save_state("discriminator", self.discriminator.state_dict(), self.discriminator_optimiser.state_dict(), epoch)
            Checkpoint.save_state("generator", self.generator.state_dict(), self.generator_optimiser.state_dict(), epoch)
            epoch += 1


if __name__ == "__main__":
    train = Train(1000, 2048, 256)
    train.train("../../data/v255ae_At_072_060000.lba", 100)
