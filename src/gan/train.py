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
import argparse
import logging
from datetime import datetime
from torch import nn, optim, Tensor, version, cuda
from gan import USE_CUDA
from gan.checkpoint import Checkpoint
from gan.data import Data
from gan.models.single_polarisation_single_frequency import Generator, Discriminator
from gan.visualise import Visualiser

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN model")
    parser.add_argument('--autoencoder', default=True,
                        help="If True, enable the autoencoder")
    parser.add_argument('-b, --batch', default=2048, type=int,
                        help="batch size used for training GAN model")
    parser.add_argument('--fft', default=True,
                        help="Perform an FFT on the data and save the FFT output")
    parser.add_argument('--samples', type=int, default=1000000000 // 2,
                        help="Number of samples to read. If using --fft, this will round to the next multiple of --fft_size")  # can't load 16gb of data in you fool
    parser.add_argument('--width', type=int, default=2048,
                        help="Width size for Discriminator and Generator")
    return vars(parser.parse_args())


SAMPLE_SIZE = 1024  # 1024 signal samples to train on
TRAINING_BATCH_SIZE = 100
TRAINING_BATCHES = 10000

USE_ANGLE_ABS = False  # Convert from real, imag input to abs, angle
# if true, add dropout to the inputs before passing them into the network
ADD_DROPOUT = True
ADD_NOISE = False  # if true, add noise to the inputs before passing them into the network

print("Searching for CUDA device...")
num_device = cuda.device_count()
if num_device > 0:
    USE_CUDA = True
    print("Found", num_device, "CUDA device(s)")
    print("Checking capability...")
    if cuda.get_device_capability(0)[0] > 3:
        USE_CUDA = True
        print("Passes, CUDA enabled with version: ", version.cuda)
    else:
        USE_CUDA = False
        print("Device is too old, switched to CPU")
else:
    print("No CUDA device found, use CPU")
    USE_CUDA = False


class Train(object):

    def __init__(self, samples, width, batch_size, fft):

        self.samples = samples
        self.width = width
        self.batch_size = batch_size
        self.use_fft = fft

        # Ensure the checkpoint directories exist
        Checkpoint.create_directory("discriminator")
        Checkpoint.create_directory("generator")

        LOG.info("Creating models...")
        self._discriminator = Discriminator(width)
        self._generator = Generator(width)

        self.noise_width = self._generator.get_noise_width()

        if USE_CUDA:
            # self.discriminator = self._discriminator.cuda()
            self.generator = self._generator.cuda()
            LOG.info("Using CUDA")
        else:
            LOG.info("Using CPU")
            self.discriminator = self._discriminator
            self.generator = self._generator

        # self.discriminator_optimiser = optim.Adam(self.discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))
        self.generator_optimiser = optim.Adam(
            self.generator.parameters(), lr=0.005, betas=(0.5, 0.999))

    def train_discriminator_autoencoder(self, filename, max_epochs):
        LOG.info("Training generator as autoencoder...")

        try:
            epoch = Checkpoint.try_restore(
                "generator", self.generator, self.generator_optimiser)
            if epoch is None:
                epoch = 0
                LOG.info("No generator checkpoint to restore")
            else:
                LOG.info("Restored checkpoint at epoch {0}".format(epoch))
        except Exception as e:
            LOG.error("Failed to restore generator {0}".format(e))
            epoch = 0

        self._generator.set_autoencoder(True)
        self._train_discriminator_autoencoder(filename, epoch, max_epochs)

        LOG.info("Training complete, saving final model state")
        Checkpoint.create_directory("generator_complete")
        Checkpoint.save_state("generator_complete", self.generator.state_dict(
        ), self.generator_optimiser.state_dict(), -1)

    def _train_discriminator_autoencoder(self, filename, start_epoch, max_epochs):
        criterion = nn.SmoothL1Loss()

        LOG.info("Loading data...")
        data_loader = Data(filename, self.samples, self.width,
                           self.noise_width, self.batch_size, use_angle_abs=USE_ANGLE_ABS)
        vis = Visualiser(os.path.join(
            os.path.splitext(filename)[0], str(datetime.now())))
        epoch = start_epoch

        while epoch < max_epochs:
            for step, (data, _, _) in enumerate(data_loader):
                data_cuda = data.cuda()
                if ADD_DROPOUT:
                    # Drop out parts of the input, but compute loss on the full input.
                    out = self.generator(nn.functional.dropout(data_cuda, 0.5))
                else:
                    out = self.generator(data_cuda)
                loss = criterion(out.cpu(), data)
                self.generator.zero_grad()
                loss.backward()
                self.generator_optimiser.step()

                vis.step_autoencoder(loss.item())

                if step % 5 == 0:
                    # Report data and save checkpoint
                    fmt = "Epoch [{0}/{1}], Step[{2}], loss: {3:.4f}"
                    LOG.info(fmt.format(epoch + 1, max_epochs, step, loss))

            Checkpoint.save_state("generator", self.generator.state_dict(
            ), self.generator_optimiser.state_dict(), epoch)
            vis.plot_training(epoch)
            data, _, _ = iter(data_loader).__next__()
            vis.test_autoencoder(epoch, self.generator, data_cuda)
            epoch += 1

    def train(self, filename, max_epochs):
        # Try restoring previous model state if it exists
        LOG.info("Attempting checkpoint restore...")

        epoch = 0
        try:
            discriminator_epoch = Checkpoint.try_restore(
                "discriminator", self.discriminator, self.discriminator_optimiser)
            generator_epoch = Checkpoint.try_restore(
                "generator", self.generator, self.generator_optimiser)
            if discriminator_epoch != generator_epoch:
                LOG.error(
                    "Discriminator and generator checkpoints out of sync. Ignoring")
            elif discriminator_epoch is not None:
                epoch = discriminator_epoch
            LOG.info("Restored checkpoint at epoch {0}".format(epoch))
        except Exception as e:
            LOG.error(
                "Failed to restore discriminator and generator {0}".format(e))

        self._generator.set_autoencoder(False)
        self._train(filename, epoch, max_epochs)

        # Final save after training complete
        LOG.info("Training complete, saving final model state")
        Checkpoint.create_directory("discriminator_complete")
        Checkpoint.create_directory("generator_complete")
        Checkpoint.save_state("discriminator_complete", self.discriminator.state_dict(
        ), self.discriminator_optimiser.state_dict(), -1)
        Checkpoint.save_state("generator_complete", self.generator.state_dict(
        ), self.generator_optimiser.state_dict(), -1)

    def _train(self, filename, start_epoch, max_epochs):
        real_labels = None
        fake_labels = None

        criterion = nn.BCELoss()
        epoch = start_epoch

        LOG.info("Loading data...")
        data_loader = Data(filename, self.samples, self.width,
                           self.noise_width, self.batch_size)
        vis = Visualiser(os.path.join(
            os.path.splitext(filename)[0], str(datetime.now())))

        # Training loop
        LOG.info("Training start")
        while epoch < max_epochs:
            for step, (data, noise1, noise2) in enumerate(data_loader):
                batch_size = data.size(0)
                if real_labels is None or real_labels.size(0) != batch_size:
                    real_labels = data_loader.generate_labels(
                        batch_size, [1.0, 0.0])
                if fake_labels is None or fake_labels.size(0) != batch_size:
                    fake_labels = data_loader.generate_labels(
                        batch_size, [0.0, 1.0])

                # todo: Track training performance over time. Show random sample of input + what the generator created.

                # ============= Train the discriminator =============
                # Pass real noise through first - ideally the discriminator will return [1, 0]
                d_output_real = self.discriminator(data)
                # Pass generated noise through - ideally the discriminator will return [0, 1]
                d_output_fake1 = self.discriminator(self.generator(noise1))

                # Determine the loss of the discriminator by adding up the real and fake loss and backpropagate
                # How good the discriminator is on real input
                d_loss_real = criterion(d_output_real, real_labels)
                # How good the discriminator is on fake input
                d_loss_fake = criterion(d_output_fake1, fake_labels)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                self.discriminator.zero_grad()
                d_loss.backward()
                self.discriminator_optimiser.step()

                # =============== Train the generator ===============
                # Pass in fake noise to the generator and get it to generate "real" noise
                # Judge how good this noise is with the discriminator
                d_output_fake2 = self.discriminator(self.generator(noise2))

                # Determine the loss of the generator using the discriminator and backpropagate
                g_loss = criterion(d_output_fake2, real_labels)
                self.discriminator.zero_grad()
                self.generator.zero_grad()
                g_loss.backward()
                self.generator_optimiser.step()

                vis.step(d_loss_real.item(), d_loss_fake.item(), g_loss.item())

                if step % 5 == 0:
                    # Report data and save checkpoint
                    fmt = "Epoch [{0}/{1}], Step[{2}], d_loss_real: {3:.4f}, d_loss_fake: {4:.4f}, g_loss: {5:.4f}"
                    LOG.info(fmt.format(epoch + 1, max_epochs, step,
                                        d_loss_real, d_loss_fake, g_loss))
                    # self.plots.generate(real, g_output_fake1, g_output_fake2, epoch)

            Checkpoint.save_state("discriminator", self.discriminator.state_dict(
            ), self.discriminator_optimiser.state_dict(), epoch)
            Checkpoint.save_state("generator", self.generator.state_dict(
            ), self.generator_optimiser.state_dict(), epoch)
            vis.plot_training(epoch)

            data, noise1, noise2 = iter(data_loader).__next__()
            vis.test(epoch, self.discriminator, self.generator, noise1, data)
            epoch += 1


if __name__ == "__main__":
    args = parse_args()

    autoencoder, samples = args['autoencoder'], args['samples']
    width, fft, batch = args['width'], args['fft'], args['batch']
    if fft:
        # 1000001536, next a largest multiple of width
        samples = samples - (samples % width) + width
        width *= 2  # Double width for fft (real + imag values)

    train = Train(samples // width, width, batch, fft)
    if autoencoder:
        #train.train_discriminator_autoencoder("../../data/At_c0p0_c0_p0_s1000000000_fft4096.hdf5", 50)
        train.train_discriminator_autoencoder(
            "../../data/At_c0_p0_s1000000000_fft2048.hdf5", 50)
    else:
        train.train("../../data/At_c0p0_c0_p0_s1000000000_fft4096.hdf5", 50)
