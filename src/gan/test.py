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
from gan.data import get_data_loaders
from gan.data_fft import get_data_loaders_fft
from gan.model import get_models
from gan.checkpoint import Checkpoint

SAMPLE_SIZE = 1024


class TestFFT(object):
    def __init__(self):
        self.real, self.fake, _ = get_data_loaders_fft("test.hdf5", 1, 5, use_cuda=True)
        self.train_real, self.train_fake, _ = get_data_loaders_fft("train.hdf5", 1, 5, use_cuda=True)

    def _print_results(self, discriminator, real, fake, tag):
        fake_noise_outputs = map(lambda x: discriminator(x), fake)
        real_noise_outputs = map(lambda x: discriminator(x), real)

        print("{0} Expected fake: 0, 1".format(tag))
        for index, category in enumerate(fake_noise_outputs):
            print("Fake Noise {0}: {1}".format(index, category))

        print("\n{0} Expected real: 1, 0".format(tag))
        for index, category in enumerate(real_noise_outputs):
            print("Real Noise {0}: {1}".format(index, category))

    def __call__(self, discriminator):
        discriminator.eval()
        self._print_results(discriminator, self.real, self.fake, "Test")
        self._print_results(discriminator, self.train_real, self.train_fake, "Train")
        discriminator.train()


def test(use_cuda=True):
    real_noise, fake_noise1, fake_noise2 = get_data_loaders(20, 1, SAMPLE_SIZE, use_cuda)
    discriminator, generator = get_models(SAMPLE_SIZE)

    Checkpoint.try_restore("discriminator_complete", discriminator, optimiser=None)
    Checkpoint.try_restore("generator_complete", generator, optimiser=None)

    fake_noise_outputs = map(lambda x: discriminator(x), fake_noise1)
    real_noise_outputs = map(lambda x: discriminator(x), real_noise)

    print("Expected fake: 0, 1")
    for index, category in enumerate(fake_noise_outputs):
        print("Fake Noise {0}: {1}".format(index, category))

    print("\nExpected real: 1, 0")
    for index, category in enumerate(real_noise_outputs):
        print("Real Noise {0}: {1}".format(index, category))


if __name__ == "__main__":
    test(use_cuda=False)