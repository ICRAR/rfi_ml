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
"""
Generates visualisations and plots for GAN and autoencoder training.
"""

import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Union, List

from .jobs import JobQueue

LOG = logging.getLogger(__name__)


class PdfPlotter(object):
    def __init__(self, filename, split=False):
        """
        Creates PDF plots.
        ```python
        with PdfPlotter("plots.pdf") as pdf:
            pdf.plot_output(data, "Some data")
        ```

        Parameters
        ----------
        filename : Union[str, list]
            The filename to save the PDF to
        split : boolean

        """
        self.filename = filename
        self.pdf = None
        self.split = split

    def __enter__(self):
        self.pdf = PdfPages(self.filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        LOG.info("Writing PDF to {0}".format(self.filename))
        self.pdf.close()
        self.pdf = None
        return self

    def plot_learning(self, data: List[float], title: Union[str, list]):
        """
        Plot the learning of a model to the PDF file.

        Parameters
        ----------
        data
            The learning data
        title : str
            The plot title
        """
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.plot(data)
        self.pdf.savefig()
        fig.clear()
        plt.close(fig)

    def plot_output(self, data, title: Union[str, list]):
        """
        Plot the output of a model to the PDF file.

        Parameters
        ----------
        data
            The output data
        title : Union[str, list]
            The plot title
        """

        def plot_single(data, title):
            plt.title(title)
            plt.plot(data)

        def plot_split(data, title):
            if self.split:
                plt.subplot(1, 2, 1)
                plot_single(data[0], title[0])
                plt.subplot(1, 2, 2)
                plot_single(data[1], title[1])
                plt.tight_layout()
            else:
                plot_single(data, title)

        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.xlabel('Sample')
        plt.ylabel('Value')
        if type(data) == list:
            for d in data:
                plot_split(d, title)
        else:
            plot_split(data, title)
        self.pdf.savefig()
        fig.clear()
        plt.close(fig)


class AutoEncoderTest(object):
    def __init__(self, directory: str, epoch: int, out, real):
        """
        A job to plot the test output from the autoencoder

        Parameters
        ----------
        directory : str
            The directory to save the plot to.
        epoch : int
            The training epoch
        out
            The autoencoder's output
        real
            The real data
        """
        self.directory = directory
        self.epoch = epoch
        self.out = out
        self.real = real

    def __call__(self, *args, **kwargs):
        with PdfPlotter(os.path.join(self.directory, "{0}_plots.pdf".format(self.epoch)), split=True) as pdf:
            for i in range(min(5, self.out.shape[0])):
                base = "Autoencoder Output {0}".format(i)
                pdf.plot_output(self.out[i], ["{0}: absolute".format(base), "{0}: angle".format(base)])
                base = "Real Output {0}".format(i)
                pdf.plot_output(self.real[i], ["{0}: absolute".format(base,), "{0}: angle".format(base)])
                base = "Output Real Comparison {0}".format(i)
                pdf.plot_output([self.real[i], self.out[i]], ["{0}: absolute".format(base), "{0}: angle".format(base)])


class GANTest(object):
    def __init__(self, directory: str, epoch: int, gen_out, real_out, discriminator_out, discriminator_real):
        """
        A job to plot the output from the generator and discriminator of the GAN.

        Parameters
        ----------
        directory : str
            The directory to save the plot to.
        epoch : int
            The training epoch
        gen_out
            The generator's output
        real_out
            The real output
        discriminator_out
            The discriminators output
        discriminator_real
            The real output
        """
        self.directory = directory
        self.epoch = epoch
        self.gen_out = gen_out
        self.real_out = real_out
        self.discriminator_out = discriminator_out
        self.discriminator_real = discriminator_real

    def __call__(self, *args, **kwargs):
        with PdfPlotter(os.path.join(self.directory, "{0}_plots.pdf".format(self.epoch))) as pdf:
            for i in range(min(10, self.gen_out.shape[0], self.real_out.shape[0])):
                pdf.plot_output(self.gen_out[i], "Generator Output {0}".format(i))
                pdf.plot_output(self.real_out[i], "Real Data {0}".format(i))
                pdf.plot_output([self.real_out[i], self.gen_out[i]], "Combined {0}".format(i))

            with open(os.path.join(self.directory, '{0}_discriminator.txt'), 'w') as f:
                    f.write("Fake Expected (Data that came from the generator): [0]\n")
                    for i in range(self.discriminator_out.shape[0]):
                        f.write("Fake: [{:.2f}]\n".format(self.discriminator_out[i][0]))  #, self.discriminator_out[i][1]))

                    f.write("\nReal Expected (Data that came from the dataset): [1]\n")

                    for i in range(self.discriminator_real.shape[0]):
                        f.write("Real: [{:.2f}]\n".format(self.discriminator_real[i][0]))  #, self.discriminator_real[i][1]))


class PlotLearning(object):
    def __init__(self, directory: str, epoch: int, d_loss_real: List[float], d_loss_fake: List[float], g_loss: List[float]):
        """
        A job to plot the learning of the GAN

        Parameters
        ----------
        directory : str
            The directory to save the plot to.
        epoch : int
            The training epoch
        d_loss_real : List[float]
            The loss of the discriminator on real data.
        d_loss_fake : List[float]
            The loss of the discriminator on fake data.
        g_loss : List[float]
            The loss of the generator.
        """
        self.directory = directory
        self.epoch = epoch
        self.d_loss_real = d_loss_real
        self.d_loss_fake = d_loss_fake
        self.g_loss = g_loss

    def __call__(self, *args, **kwargs):
        with PdfPlotter(os.path.join(self.directory, "{0}_training.pdf".format(self.epoch))) as pdf:
            if len(self.d_loss_real) > 0:
                pdf.plot_learning(self.d_loss_real, "Discriminator Loss Real")
            if len(self.d_loss_fake) > 0:
                pdf.plot_learning(self.d_loss_fake, "Discriminator Loss Fake")
            if len(self.g_loss) > 0:
                pdf.plot_learning(self.g_loss, "Generator Loss")


class Visualiser(object):
    def __init__(self, base_directory: str):
        """
        Handle the creation of visualisation plots for each training epoch of GAN or autoencoder training
        ```python
        with Visualiser(vis_path) as vis:
            epoch = 0
            while(epoch < 10):
                step = 0
                while(step < 100):
                    # perform training step
                    # ...
                    # add training losses to visualiser
                    vis.step(d_loss_real, d_loss_fake, g_loss)
                    step += 1

                # plot training at the end of the epoch
                vis.plot_training(epoch)
                epoch += 1

        ```

        Parameters
        ----------
        base_directory : str
            The directory to store the plots into
        """
        self.d_loss_real = []
        self.d_loss_fake = []
        self.g_loss = []
        self.base_directory = base_directory

    def __enter__(self):
        self.queue = JobQueue(num_processes=1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.queue.join()

    def _get_directory(self):
        os.makedirs(self.base_directory, exist_ok=True)
        return self.base_directory

    def step(self, d_loss_real: float, d_loss_fake: float, g_loss: float):
        """
        Add loss data for a single GAN training step

        Parameters
        ----------
        d_loss_real : float
            The discriminator's loss on real data.
        d_loss_fake : float
            The discriminator's loss on fake data.
        g_loss : float
            The generator's loss
        """
        self.d_loss_real.append(d_loss_real)
        self.d_loss_fake.append(d_loss_fake)
        self.g_loss.append(g_loss)

    def step_autoencoder(self, loss : float):
        """
        Add loss for a single autoencoder training step

        Parameters
        ----------
        loss : float
            The loss
        """
        self.g_loss.append(loss)

    def test(self, epoch: int, discriminator, generator, noise, real):
        """
        Test the discriminator and generator on the provided noise and real data and generate plots for the test.

        Parameters
        ----------
        epoch : int
            The current epoch
        discriminator
            The discriminator
        generator
            The generator
        noise
            The noise data
        real
            The real data
        """
        generator.eval()
        discriminator.eval()
        out = generator(noise)
        self.queue.submit(GANTest(directory=self._get_directory(),
                                  epoch=epoch,
                                  gen_out=out.cpu().data.numpy(),
                                  real_out=real.cpu().data.numpy(),
                                  discriminator_out=discriminator(out).cpu().data.numpy(),
                                  discriminator_real=discriminator(real).cpu().data.numpy()))
        generator.train()
        discriminator.train()

    def test_autoencoder(self, epoch: int, autoencoder, real):
        """
        Test the autoencoder on the provided data.

        Parameters
        ----------
        epoch : int
            The current epoch
        autoencoder
            The autoencoder
        real
            The real data to test on
        """
        autoencoder.eval()
        self.queue.submit(AutoEncoderTest(directory=self._get_directory(),
                                          epoch=epoch,
                                          out=autoencoder(real[:10]).cpu().data.numpy(),
                                          real=real[:10].cpu().data.numpy()))
        autoencoder.train()

    def plot_training(self, epoch: int):
        """
        Plot the current training accumulated with `Visualiser.step`.

        Parameters
        ----------
        epoch : int
            The current epoch
        """
        self.queue.submit(PlotLearning(directory=self._get_directory(),
                                       epoch=epoch,
                                       d_loss_real=self.d_loss_real,
                                       d_loss_fake=self.d_loss_fake,
                                       g_loss=self.g_loss))
