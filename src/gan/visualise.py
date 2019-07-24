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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from jobs import JobQueue

LOG = logging.getLogger(__name__)


class PdfPlotter(object):
    def __init__(self, filename, split=False):
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

    def plot_learning(self, data, title):
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.plot(data)
        self.pdf.savefig()
        fig.clear()
        plt.close(fig)

    def plot_output(self, data, title):

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
    def __init__(self, directory, epoch, out, real):
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
    def __init__(self, directory, epoch, gen_out, real_out, discriminator_out, discriminator_real):
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
    def __init__(self, directory, epoch, d_loss_real, d_loss_fake, g_loss):
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
    def __init__(self, base_directory):
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

    def step(self, d_loss_real, d_loss_fake, g_loss):
        self.d_loss_real.append(d_loss_real)
        self.d_loss_fake.append(d_loss_fake)
        self.g_loss.append(g_loss)

    def step_autoencoder(self, loss):
        self.g_loss.append(loss)

    def test(self, epoch, discriminator, generator, noise, real):
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

    def test_autoencoder(self, epoch, generator, real):
        generator.eval()
        self.queue.submit(AutoEncoderTest(directory=self._get_directory(),
                                          epoch=epoch,
                                          out=generator(real[:10]).cpu().data.numpy(),
                                          real=real[:10].cpu().data.numpy()))
        generator.train()

    def plot_training(self, epoch):
        self.queue.submit(PlotLearning(directory=self._get_directory(),
                                       epoch=epoch,
                                       d_loss_real=self.d_loss_real,
                                       d_loss_fake=self.d_loss_fake,
                                       g_loss=self.g_loss))
