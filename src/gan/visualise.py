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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Visualiser(object):
    def __init__(self, base_directory):
        self.d_loss_real = []
        self.d_loss_fake = []
        self.g_loss = []
        self.base_directory = base_directory
        self.directory = None

    def _plot_learning(self, data, title):
        fig = plt.figure(figsize=(16, 9), dpi=80)
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.plot(data)
        plt.savefig(os.path.join(self.directory, title))
        fig.clear()
        plt.close(fig)

    def _plot_output(self, data, title):
        pass

    def step(self, d_loss_real, d_loss_fake, g_loss):
        self.d_loss_real.append(d_loss_real)
        self.d_loss_fake.append(d_loss_fake)
        self.g_loss.append(g_loss)

    def test(self, discriminator, generator, noise, real):
        generator.eval()
        discriminator.eval()

        with open('discriminator.txt', 'w') as f:
            out = generator(noise)
            # todo: Get this outputting plots the generator creates, and numbers the discriminator outputs
            #self._plot_output(out, "Generator {0}".format(i))
            discriminator_out = discriminator(out)
            f.write("Fake Expected: [0, 1]\n")
            for i in range(discriminator_out.shape[0]):
                f.write("Fake: [{:.2f}, {:.2f}]\n".format(discriminator_out[i][0].item(), discriminator_out[i][1].item()))

            f.write("\nReal Expected: [1, 0]\n")
            discriminator_out = discriminator(real)
            for i in range(discriminator_out.shape[0]):
                f.write("Real: [{:.2f}, {:.2f}]\n".format(discriminator_out[i][0].item(), discriminator_out[i][1].item()))

        generator.train()
        discriminator.train()

    def plot_training(self, epoch):
        self.directory = os.path.join(self.base_directory, "{0}".format(epoch))
        os.makedirs(self.directory, exist_ok=True)

        self._plot_learning(self.d_loss_real, "Discriminator Loss Real")
        self._plot_learning(self.d_loss_fake, "Discriminator Loss Fake")
        self._plot_learning(self.g_loss, "Generator Loss")

