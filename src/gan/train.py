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

import argparse
import logging
import os
import subprocess
from datetime import datetime

import torch
from torch import nn, optim, version

from checkpoint import Checkpoint
from config import Config
from data import Data
from models.autoencoder import Autoencoder
from models.discriminator import Discriminator
from models.generator import Generator
from visualise import Visualiser

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger(__name__)


class Train(object):
    """
    Main GAN trainer. Responsible for training the GAN and pre-training the generator autoencoder.
    """

    def __init__(self, config):
        """
        Construct a new GAN trainer
        :param Config config: The parsed network configuration.
        """
        self.config = config

        LOG.info("CUDA version: {0}".format(version.cuda))
        LOG.info("Creating data loader from path {0}".format(config.FILENAME))

        self.data_loader = Data(config.FILENAME,
                                config.BATCH_SIZE,
                                horizontal_concatenate=True,            # Concat abs and angles together into one big array for each input
                                max_inputs=config.MAX_SAMPLES,          # Max inputs per polarisation and frequency
                                normalise=config.NORMALISE)             # Normalise inputs

        shape = self.data_loader.get_input_shape()
        LOG.info("Creating models with input shape {0}".format(shape))

        width = shape[0]
        self._autoencoder = Autoencoder(width)
        self._discriminator = Discriminator(width)
        self._generator = Generator(width, width)

        if config.USE_CUDA:
            LOG.info("Using CUDA")
            self.autoencoder = self._autoencoder.cuda()
            self.discriminator = self._discriminator.cuda()
            self.generator = self._generator.cuda()
        else:
            LOG.info("Using CPU")
            self.autoencoder = self._autoencoder
            self.discriminator = self._discriminator
            self.generator = self._generator

    def check_requeue(self, epochs_complete):
        """
        Check and re-queue the training script if it has completed the desired number of training epochs per session
        :param int epochs_complete: Number of epochs completed
        :return: True if the script has been requeued, False if not
        :rtype bool
        """
        if self.config.REQUEUE_EPOCHS > 0:
            if epochs_complete >= self.config.REQUEUE_EPOCHS:
                # We've completed enough epochs for this instance. We need to kill it and requeue
                LOG.info("REQUEUE_EPOCHS of {0} met, calling REQUEUE_SCRIPT".format(self.config.REQUEUE_EPOCHS))
                subprocess.call(self.config.REQUEUE_SCRIPT, shell=True, cwd=os.path.dirname(self.config.REQUEUE_SCRIPT))
                return True  # Requeue performed
        return False  # No requeue needed

    def load_state(self, checkpoint, module, optimiser=None):
        """
        Load the provided checkpoint into the provided module and optimiser.
        This function checks whether the load threw an exception and logs it to the user.
        :param Checkpoint checkpoint: The checkpoint to load
        :param module: The pytorch module to load the checkpoint into.
        :param optimiser: The pytorch optimiser to load the checkpoint into.
        :return: None if the load failed, int number of epochs in the checkpoint if load succeeded
        """
        try:
            module.load_state_dict(checkpoint.module_state)
            if optimiser is not None:
                optimiser.load_state_dict(checkpoint.optimiser_state)
            return checkpoint.epoch
        except RuntimeError as e:
            LOG.exception("Error loading module state. This is most likely an input size mismatch. Please delete the old module saved state, or change the input size")
            return None

    def close(self):
        """
        Close the data loader used by the trainer.
        """
        self.data_loader.close()

    def generate_labels(self, num_samples, pattern):
        """
        Generate labels for the discriminator.
        :param int num_samples: Number of input samples to generate labels for.
        :param list pattern: Pattern to generator. Should be either [1, 0], or [0, 1]
        :return: New labels for the discriminator
        """
        var = torch.FloatTensor([pattern] * num_samples)
        return var.cuda() if self.config.USE_CUDA else var

    def _train_autoencoder(self):
        """
        Main training loop for the autencoder.
        This function will return False if:
        - Loading the autoencoder succeeded, but the NN model did not load the state dicts correctly.
        - The script needs to be re-queued because the NN has been trained for REQUEUE_EPOCHS
        :return: True if training was completed, False if training needs to continue.
        :rtype bool
        """

        criterion = nn.SmoothL1Loss()
        optimiser = optim.Adam(self.generator.parameters(), lr=0.00003, betas=(0.5, 0.999))

        # Load saved autoencoder state
        checkpoint = Checkpoint(self.config.CHECKPOINT_DIRECTORY, "autoencoder")
        epoch = 0
        if checkpoint.load():
            epoch = self.load_state(checkpoint, self.autoencoder, optimiser)
            if epoch is not None and epoch >= self.config.MAX_AUTOENCODER_EPOCHS:
                LOG.info("Autoencoder already trained")
                return True
            else:
                LOG.info("Autoencoder training beginning from epoch {0}".format(epoch))
        else:
            LOG.info('Autoencoder checkpoint not found. Training from start')

        # Train autoencoder
        self._autoencoder.set_mode(Autoencoder.Mode.AUTOENCODER)

        vis_path = os.path.join(
            self.config.RESULT_DIRECTORY,
            os.path.splitext(self.config.FILENAME)[0], "autoencoder", str(datetime.now())
        )
        with Visualiser(vis_path) as vis:
            epochs_complete = 0
            while epoch < self.config.MAX_AUTOENCODER_EPOCHS:

                if self.check_requeue(epochs_complete):
                    return False  # Requeue needed and training not complete

                for step, (data, _, _) in enumerate(self.data_loader):
                    if self.config.USE_CUDA:
                        data = data.cuda()

                    if self.config.ADD_DROPOUT:
                        # Drop out parts of the input, but compute loss on the full input.
                        out = self.autoencoder(nn.functional.dropout(data, 0.5))
                    else:
                        out = self.autoencoder(data)

                    loss = criterion(out.cpu(), data.cpu())
                    self.autoencoder.zero_grad()
                    loss.backward()
                    optimiser.step()

                    vis.step_autoencoder(loss.item())

                    # Report data and save checkpoint
                    fmt = "Epoch [{0}/{1}], Step[{2}/{3}], loss: {4:.4f}"
                    LOG.info(fmt.format(epoch + 1, self.config.MAX_AUTOENCODER_EPOCHS, step, len(self.data_loader), loss))

                epoch += 1
                epochs_complete += 1

                checkpoint.set(self.autoencoder.state_dict(), optimiser.state_dict(), epoch).save()

                LOG.info("Plotting autoencoder progress")
                vis.plot_training(epoch)
                data, _, _ = iter(self.data_loader).__next__()
                vis.test_autoencoder(epoch, self.autoencoder, data.cuda())

        LOG.info("Autoencoder training complete")
        return True  # Training complete

    def _train_gan(self):
        """
        TODO: Add in autoencoder to perform dimensionality reduction on data
        TODO: Not working yet - trying to work out good autoencoder model first
        :return:
        """

        criterion = nn.BCELoss()
        discriminator_optimiser = optim.Adam(self.discriminator.parameters(), lr=0.003, betas=(0.5, 0.999))
        discriminator_scheduler = optim.lr_scheduler.LambdaLR(discriminator_optimiser, lambda epoch: 0.97 ** epoch)
        generator_optimiser = optim.Adam(self.generator.parameters(), lr=0.003, betas=(0.5, 0.999))
        generator_scheduler = optim.lr_scheduler.LambdaLR(generator_optimiser, lambda epoch: 0.97 ** epoch)

        # Load discriminator state in
        discriminator_checkpoint = Checkpoint(self.config.CHECKPOINT_DIRECTORY, "discriminator")
        discriminator_epoch = 0
        if discriminator_checkpoint.load():
            discriminator_epoch = self.load_state(discriminator_checkpoint, self.discriminator, discriminator_optimiser)
        else:
            LOG.info('Discriminator checkpoint not found')

        # Load generator state in
        generator_checkpoint = Checkpoint(self.config.CHECKPOINT_DIRECTORY, "generator")
        generator_epoch = 0
        if generator_checkpoint.load():
            generator_epoch = self.load_state(generator_checkpoint, self.generator, generator_optimiser)
        else:
            LOG.info('Generator checkpoint not found')

        # If discriminator / generator states don't match, train from the start.
        if discriminator_epoch is None or generator_epoch is None:
            epoch = 0
            LOG.info("Discriminator or generator failed to load, training from start")
        else:
            epoch = min(generator_epoch, discriminator_epoch)
            LOG.info("Generator loaded at epoch {0}".format(generator_epoch))
            LOG.info("Discriminator loaded at epoch {0}".format(discriminator_epoch))
            LOG.info("Training from lowest epoch {0}".format(epoch))

        vis_path = os.path.join(
            self.config.RESULT_DIRECTORY,
            os.path.splitext(self.config.FILENAME)[0], "gan", str(datetime.now())
        )
        with Visualiser(vis_path) as vis:
            real_labels = None  # all 1s
            fake_labels = None  # all 0s
            epochs_complete = 0
            while epoch < self.config.MAX_EPOCHS:

                if self.check_requeue(epochs_complete):
                    return  # Requeue needed and training not complete

                for step, (data, noise1, noise2) in enumerate(self.data_loader):
                    batch_size = data.size(0)
                    if real_labels is None or real_labels.size(0) != batch_size:
                        real_labels = self.generate_labels(batch_size, [1.0])
                    if fake_labels is None or fake_labels.size(0) != batch_size:
                        fake_labels = self.generate_labels(batch_size, [0.0])

                    if self.config.USE_CUDA:
                        data = data.cuda()
                        noise1 = noise1.cuda()
                        noise2 = noise2.cuda()

                    # ============= Train the discriminator =============
                    # Pass real noise through first - ideally the discriminator will return 1 #[1, 0]
                    d_output_real = self.discriminator(data)
                    # Pass generated noise through - ideally the discriminator will return 0 #[0, 1]
                    d_output_fake1 = self.discriminator(self.generator(noise1))

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
                    d_output_fake2 = self.discriminator(self.generator(noise2))

                    # Determine the loss of the generator using the discriminator and backpropagate
                    g_loss = criterion(d_output_fake2, real_labels)
                    self.discriminator.zero_grad()
                    self.generator.zero_grad()
                    g_loss.backward()
                    generator_optimiser.step()

                    vis.step(d_loss_real.item(), d_loss_fake.item(), g_loss.item())

                    # Report data and save checkpoint
                    fmt = "Epoch [{0}/{1}], Step[{2}/{3}], d_loss_real: {4:.4f}, d_loss_fake: {5:.4f}, g_loss: {6:.4f}"
                    LOG.info(fmt.format(epoch + 1, self.config.MAX_EPOCHS, step + 1, len(self.data_loader), d_loss_real, d_loss_fake, g_loss))

                epoch += 1
                epochs_complete += 1

                discriminator_checkpoint.set(self.discriminator.state_dict(), discriminator_optimiser.state_dict(), epoch).save()
                generator_checkpoint.set(self.generator.state_dict(), generator_optimiser.state_dict(), epoch).save()
                vis.plot_training(epoch)

                data, noise1, _ = iter(self.data_loader).__next__()
                if self.config.USE_CUDA:
                    data = data.cuda()
                    noise1 = noise1.cuda()
                vis.test(epoch, self.discriminator, self.generator, noise1, data)

                generator_scheduler.step(epoch)
                discriminator_scheduler.step(epoch)

                LOG.info("Learning rates: d {0} g {1}".format(
                    discriminator_optimiser.param_groups[0]["lr"],
                    generator_optimiser.param_groups[0]["lr"]
                ))

        LOG.info("GAN Training complete")

    def __call__(self):
        """
        Main training loop for the GAN.
        The training process is interruptable; the model and optimiser states are saved to disk each epoch, and the
        latest states are restored when the trainer is resumed.

        If the script is not able to load the generator's saved state, it will attempt to load the pre-trained generator
        autoencoder from the generator_decoder_complete checkpoint (if it exists). If this also fails, the generator is
        pre-trained as an autoencoder. This training is also interruptable, and will produce the
        generator_decoder_complete checkpoint on completion.

        On successfully restoring generator and discriminator state, the trainer will proceed from the earliest restored
        epoch. For example, if the generator is restored from epoch 7 and the discriminator is restored from epoch 5,
        training will proceed from epoch 5.

        Visualisation plots are produces each epoch and stored in
        /path_to_input_file_directory/{gan/generator_auto_encoder}/{timestamp}/{epoch}

        Each time the trainer is run, it creates a new timestamp directory using the current time.
        """

        # Load the autoencoder, and train it if needed.
        #if not self._train_autoencoder():
            # Autoencoder training incomplete
        #    return

        self._train_gan()


def parse_args():
    """
    Parse command line arguments for the trainer.
    :return: Parsed command line arguments as a dict.
    :rtype dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Path to the config file')
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    train = Train(Config(args['config_file']))
    train()
    train.close()
