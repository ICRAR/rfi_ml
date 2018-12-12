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
from datetime import datetime
from torch import nn, optim, version
from gan.checkpoint import Checkpoint
from gan.data import Data
from gan.models.single_polarisation_single_frequency import Generator, Discriminator
from gan.visualise import Visualiser
from gan.config import Config

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)

print(version.cuda)


class Train(object):

    def __init__(self, config):
        self.config = config

        LOG.info("Creating models...")
        self._discriminator = Discriminator(config)
        self._generator = Generator(config)

        self.noise_width = self._generator.get_noise_width()
        self.data_loader = None

        if config.USE_CUDA:
            self.discriminator = self._discriminator.cuda()
            self.generator = self._generator.cuda()
            LOG.info("Using CUDA")
        else:
            LOG.info("Using CPU")
            self.discriminator = self._discriminator
            self.generator = self._generator

    def _train_generator_autoencoder(self):
        optimiser = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        criterion = nn.SmoothL1Loss()
        epoch = 0

        generator_decoder_checkpoint = Checkpoint("generator_autoencoder")
        if generator_decoder_checkpoint.load():
            self.generator.load_state_dict(generator_decoder_checkpoint.module_state)
            optimiser.load_state_dict(generator_decoder_checkpoint.optimiser_state)
            epoch = generator_decoder_checkpoint.epoch
            LOG.info("Successfully loaded generator autoencoder state at epoch {0}".format(epoch))
        else:
            LOG.info("Failed to load generator autoencoder state. Training from start")

        self._generator.set_autoencoder(True)

        LOG.info("Loading data...")
        self.data_loader = Data(self.config, self.noise_width)

        vis_path = os.path.join(os.path.splitext(self.config.FILENAME)[0], "generator_autoencoder", str(datetime.now()))
        with Visualiser(vis_path) as vis:
            while epoch < self.config.MAX_GENERATOR_AUTOENCODER_EPOCHS:
                for step, (data, _, _) in enumerate(self.data_loader):
                    data_cuda = data.cuda()
                    if self.config.ADD_DROPOUT:
                        # Drop out parts of the input, but compute loss on the full input.
                        out = self.generator(nn.functional.dropout(data_cuda, 0.5))
                    else:
                        out = self.generator(data_cuda)

                    loss = criterion(out.cpu(), data)
                    self.generator.zero_grad()
                    loss.backward()
                    optimiser.step()

                    vis.step_autoencoder(loss.item())

                    if step % 5 == 0:
                        # Report data and save checkpoint
                        fmt = "Epoch [{0}/{1}], Step[{2}], loss: {3:.4f}"
                        LOG.info(fmt.format(epoch + 1, self.config.MAX_GENERATOR_AUTOENCODER_EPOCHS, step, loss))

                Checkpoint("generator_autoencoder", self.generator.state_dict(), optimiser.state_dict(), epoch).save()

                vis.plot_training(epoch)
                data, _, _ = iter(self.data_loader).__next__()
                vis.test_autoencoder(epoch, self.generator, data.cuda())
                epoch += 1

        LOG.info("Saving final generator decoder state")
        Checkpoint("generator_decoder_complete", self.generator.decoder.state_dict()).save()

    def __call__(self):
        # When training the GAN, we only want to use the decoder part of the generator.
        generator_optimiser = optim.Adam(self.generator.decoder.parameters(), lr=0.0022, betas=(0.5, 0.999))
        discriminator_optimiser = optim.Adam(self.discriminator.parameters(), lr=0.0022, betas=(0.5, 0.999))

        generator_scheduler = optim.lr_scheduler.LambdaLR(generator_optimiser, lambda epoch: 0.95 ** epoch)
        discriminator_scheduler = optim.lr_scheduler.LambdaLR(discriminator_optimiser, lambda epoch: 0.9 ** epoch)

        criterion = nn.BCELoss()

        generator_epoch = 0
        discriminator_epoch = 0

        # Load in the GAN generator state.
        # This will actually be the state of the generator decoder only.
        generator_checkpoint = Checkpoint("generator")
        if generator_checkpoint.load():
            self.generator.load_state_dict(generator_checkpoint.module_state)
            generator_optimiser.load_state_dict(generator_checkpoint.optimiser_state)
            generator_epoch = generator_checkpoint.epoch
            LOG.info("Successfully loaded generator state at epoch {0}".format(generator_epoch))
        else:
            # Failed to get GAN generator state, so try and find the final state of the
            # Generator decoder that's saved after pre-training, and load it.
            LOG.info("Failed to load generator state.")
            generator_decoder_checkpoint = Checkpoint("generator_decoder_complete")
            if generator_decoder_checkpoint.load():
                LOG.info("Successfully loaded completed generator decoder state")
                self.generator.decoder.load_state_dict(generator_decoder_checkpoint.module_state)
            else:
                # Can't find a final decoder state, so we need to train the generator as an autoencoder
                # then save the final state of the decoder.
                LOG.info("Failed to load completed generator decoder state. Training now")
                self._train_generator_autoencoder()

        discriminator_checkpoint = Checkpoint("discriminator")
        if discriminator_checkpoint.load():
            self.discriminator.load_state_dict(discriminator_checkpoint.module_state)
            discriminator_optimiser.load_state_dict(discriminator_checkpoint.optimiser_state)
            discriminator_epoch = discriminator_checkpoint.epoch
            LOG.info("Successfully loaded discriminator state at epoch {0}".format(discriminator_epoch))
        else:
            LOG.info("Failed to load discriminator state.")

        epoch = min(generator_epoch, discriminator_epoch)
        LOG.info("Generator epoch: {0}. Discriminator epoch: {1}. Proceeding from earliest epoch: {2}"
                 .format(generator_epoch, discriminator_epoch, epoch))

        if self.data_loader is None:
            # Data might have been loaded by the generator autoencoder training loop
            LOG.info("Loading data...")
            self.data_loader = Data(self.config, self.noise_width)

        vis_path = os.path.join(os.path.splitext(self.config.FILENAME)[0], "gan", str(datetime.now()))
        with Visualiser(vis_path) as vis:
            self._generator.set_autoencoder(False)
            real_labels = None
            fake_labels = None

            while epoch < self.config.MAX_EPOCHS:
                for step, (data, noise1, noise2) in enumerate(self.data_loader):
                    batch_size = data.size(0)
                    if real_labels is None or real_labels.size(0) != batch_size:
                        real_labels = self.data_loader.generate_labels(batch_size, [1.0, 0.0])
                    if fake_labels is None or fake_labels.size(0) != batch_size:
                        fake_labels = self.data_loader.generate_labels(batch_size, [0.0, 1.0])

                    data_cuda = data.cuda()
                    noise1_cuda = noise1.cuda()
                    noise2_cuda = noise2.cuda()

                    # ============= Train the discriminator =============
                    # Pass real noise through first - ideally the discriminator will return [1, 0]
                    d_output_real = self.discriminator(data_cuda)
                    # Pass generated noise through - ideally the discriminator will return [0, 1]
                    d_output_fake1 = self.discriminator(self.generator(noise1_cuda))

                    # Determine the loss of the discriminator by adding up the real and fake loss and backpropagate
                    d_loss_real = criterion(d_output_real, real_labels)  # How good the discriminator is on real input
                    d_loss_fake = criterion(d_output_fake1, fake_labels)  # How good the discriminator is on fake input
                    d_loss = d_loss_real + d_loss_fake
                    self.discriminator.zero_grad()
                    d_loss.backward()
                    discriminator_optimiser.step()

                    # =============== Train the generator ===============
                    # Pass in fake noise to the generator and get it to generate "real" noise
                    # Judge how good this noise is with the discriminator
                    d_output_fake2 = self.discriminator(self.generator(noise2_cuda))

                    # Determine the loss of the generator using the discriminator and backpropagate
                    g_loss = criterion(d_output_fake2, real_labels)
                    self.discriminator.zero_grad()
                    self.generator.zero_grad()
                    g_loss.backward()
                    generator_optimiser.step()

                    vis.step(d_loss_real.item(), d_loss_fake.item(), g_loss.item())

                    if step % 5 == 0:
                        # Report data and save checkpoint
                        fmt = "Epoch [{0}/{1}], Step[{2}], d_loss_real: {3:.4f}, d_loss_fake: {4:.4f}, g_loss: {5:.4f}"
                        LOG.info(fmt.format(epoch + 1, self.config.MAX_EPOCHS, step, d_loss_real, d_loss_fake, g_loss))


                Checkpoint("discriminator", self.discriminator.state_dict(), discriminator_optimiser, epoch).save()
                Checkpoint("generator", self.generator.decoder.state_dict(), generator_optimiser, epoch).save()
                vis.plot_training(epoch)

                data, noise1, noise2 = iter(self.data_loader).__next__()
                vis.test(epoch, self.discriminator, self.generator, noise1.cuda(), data.cuda())
                epoch += 1

                generator_scheduler.step(epoch)
                discriminator_scheduler.step(epoch)

                LOG.info("Learning rates: d {0} g {1}".format(
                    discriminator_optimiser.param_groups[0]["lr"],
                    generator_optimiser.param_groups[0]["lr"]
                ))

        LOG.info("Training complete, saving final model state")
        Checkpoint("discriminator_complete", self.discriminator.state_dict(), discriminator_optimiser.state_dict(), epoch).save()
        Checkpoint("generator_complete", self.generator.decoder.state_dict(), generator_optimiser.state_dict(), epoch).save()


if __name__ == "__main__":
    train = Train(Config("gan_config.settings"))
    train()
