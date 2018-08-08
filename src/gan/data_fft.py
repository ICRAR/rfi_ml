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

import numpy as np
import logging
import h5py
from torch.utils.data import DataLoader

LOG = logging.getLogger(__name__)


def get_data_loaders_fft(filename, batch_size, total_inputs, use_cuda):
    with h5py.File(filename, 'r') as f:
        fake_noise_data1 = DataLoader(np.copy(f["fake1"])[:total_inputs, :],
                                      batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=use_cuda,
                                      num_workers=1)

        fake_noise_data2 = DataLoader(np.copy(f["fake2"])[:total_inputs, :],
                                      batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=use_cuda,
                                      num_workers=1)

        real_noise_data = DataLoader(np.copy(f['real']['p0']['f0']).astype(np.float32)[:total_inputs, :],
                                     batch_size=batch_size,
                                     shuffle=True,
                                     pin_memory=use_cuda,
                                     num_workers=1)

    return real_noise_data, fake_noise_data1, fake_noise_data2


if __name__ == "__main__":
    real, fake1, fake2 = get_data_loaders_fft(100, 10000, 1024, True)

    for i, (r, f1, f2) in enumerate(zip(real, fake1, fake2)):
        print(r, f1, f2)
        if i == 5:
            break