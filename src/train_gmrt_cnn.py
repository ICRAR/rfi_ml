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

"""
import argparse
import logging
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.data as data

from constants import NUMBER_CHANNELS, NUMBER_OF_CLASSES
from train import test_epoch, train
from utilities import RfiData, Timer, build_data

LOGGER = logging.getLogger(__name__)
HIDDEN_LAYERS = 200


class GmrtCNN(nn.Module):
    def __init__(self, keep_probability):
        super(GmrtCNN, self).__init__()
        self.keep_probability = keep_probability
        self.conv1 = nn.Conv1d(NUMBER_CHANNELS, 50, kernel_size=3, stride=1)
        self.conv1.double()     # Force the Conv1d to use a double
        self.max_pool1 = nn.MaxPool1d(2, stride=2)
        self.conv2 = nn.Conv1d(50, 100, kernel_size=2, stride=1)
        self.conv2.double()     # Force the Conv1d to use a double
        self.max_pool2 = nn.MaxPool1d(2, stride=2)
        self.conv3 = nn.Conv1d(100, 200, kernel_size=1, stride=1)
        self.conv3.double()     # Force the Conv1d to use a double
        self.max_pool3 = nn.MaxPool1d(2, stride=1)
        self.fc1a = nn.Linear(600, 300)
        self.fc1a.double()       # Force the layer to a double
        self.fc1b = nn.Linear(300, 10)
        self.fc1b.double()       # Force the layer to a double

        self.fc2 = nn.Linear(18, 2048)
        self.fc2 = nn.Linear(8, 2048)
        self.fc2.double()       # Force the layer to use a double
        self.fc3 = nn.Linear(2048, 1024)
        self.fc3.double()       # Force the layer to use a double
        self.fc4 = nn.Linear(1024, NUMBER_OF_CLASSES)
        self.fc4.double()       # Force the layer to use a double

    def forward(self, input_data_ts, input_data_values):
        x = self.max_pool1(functional.relu(self.conv1(input_data_ts)))
        x = self.max_pool2(functional.relu(self.conv2(x)))
        x = self.max_pool3(functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = functional.dropout(x, p=self.keep_probability, training=self.training)
        x = functional.relu(self.fc1a(x))
        x = functional.relu(self.fc1b(x))

        z = torch.cat((x, input_data_values), dim=1)
        z = functional.relu(self.fc2(input_data_values))
        z = functional.relu(self.fc3(z))
        z = functional.relu(self.fc4(z))

        z = functional.tanh(z)
        return z


class GmrtLinear(nn.Module):
    def __init__(self, keep_probability, sequence_length):
        super(GmrtLinear, self).__init__()
        self.keep_probability = keep_probability
        self.input_layer_length = 6 + (sequence_length * 7)

        self.fc1 = nn.Linear(self.input_layer_length, HIDDEN_LAYERS).double()
        self.fc2 = nn.Linear(HIDDEN_LAYERS + self.input_layer_length, HIDDEN_LAYERS).double()
        self.fc3 = nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS).double()
        self.fc4 = nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS).double()
        self.fc5 = nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS).double()
        self.fc6 = nn.Linear(HIDDEN_LAYERS, NUMBER_OF_CLASSES).double()

    def forward(self, input_data_values):
        x = functional.leaky_relu(self.fc1(input_data_values))
        x = functional.leaky_relu(self.fc2(torch.cat((x, input_data_values), dim=1)))
        x = functional.dropout(x, p=self.keep_probability, training=self.training)
        x = functional.leaky_relu(self.fc3(x))
        x = functional.leaky_relu(self.fc4(x))
        x = functional.dropout(x, p=self.keep_probability, training=self.training)
        x = functional.leaky_relu(self.fc5(x))
        x = functional.leaky_relu(self.fc6(x))

        x = functional.softmax(x)
        return x


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(process)d:%(levelname)s:%(name)s:%(message)s')
    parser = argparse.ArgumentParser(description='GMRT CNN Training')
    parser.add_argument('--batch-size', type=int, default=20000, metavar='N', help='input batch size for training (default: 20000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--keep-probability', type=float, default=0.6, metavar='K', help='Dropout keep probability (default: 0.6)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--num-processes', type=int, default=4, metavar='N', help='how many training processes to use (default: 4)')
    parser.add_argument('--use-gpu', action='store_true', default=False, help='use the GPU if it is available')
    parser.add_argument('--data-path', default='./data', help='the path to the data file')
    parser.add_argument('--data-file', default='data.h5', help='the name of the data file')
    parser.add_argument('--sequence-length', type=int, default=10, help='how many elements in a sequence')
    parser.add_argument('--validation-percentage', type=int, default=10, help='amount of data used for validation')
    parser.add_argument('--training-percentage', type=int, default=80, help='amount of data used for training')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--learning-rate-decay', type=float, default=0.8, metavar='LRD', help='the initial learning rate decay rate')
    parser.add_argument('--start-learning-rate-decay', type=int, default=5, help='the epoch to start applying the LRD')
    parser.add_argument('--short_run', type=int, default=None, help='use a short run of the test data')
    parser.add_argument('--save', type=str,  default=None, help='path to save the final model')

    kwargs = vars(parser.parse_args())
    LOGGER.debug(kwargs)

    # If the have specified a seed get a random
    if kwargs['seed'] is not None:
        np.random.seed(kwargs['seed'])
    else:
        np.random.seed()

    if kwargs['use_gpu'] and torch.cuda.is_available():
        LOGGER.info('Using cuda devices: {}'.format(torch.cuda.device_count()))
        kwargs['cuda_device_count'] = torch.cuda.device_count()
        kwargs['using_gpu'] = True
    else:
        LOGGER.info('Using CPU')
        kwargs['cuda_device_count'] = 0
        kwargs['using_gpu'] = False

    # Do this first so all the data is built before we go parallel and get race conditions
    with Timer('Checking/Building data file'):
        build_data(**kwargs)

    rfi_data = RfiData(**kwargs)

    if kwargs['using_gpu']:
        # The DataParallel will distribute the model to all the available GPUs
        # model = nn.DataParallel(GmrtCNN(kwargs['keep_probability'])).cuda()
        model = nn.DataParallel(GmrtLinear(kwargs['keep_probability'], kwargs['sequence_length'])).cuda()

        # Train
        train(model, rfi_data, **kwargs)

    else:
        # This uses the HOGWILD! approach to lock free SGD
        # model = GmrtCNN(kwargs['keep_probability'])
        model = GmrtLinear(kwargs['keep_probability'], kwargs['sequence_length'])
        model.share_memory()  # gradients are allocated lazily, so they are not shared here

        processes = []
        for rank in range(kwargs['num_processes']):
            p = mp.Process(target=train, args=(model, rfi_data, rank), kwargs=kwargs)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    with Timer('Reading final test data'):
        test_loader = data.DataLoader(
            rfi_data.get_rfi_dataset('test', short_run_size=kwargs['short_run']),
            batch_size=kwargs['batch_size'],
            num_workers=1,
            pin_memory=kwargs['using_gpu'],
        )
    with Timer('Final test'):
        test_epoch(model, test_loader, kwargs['log_interval'])

    if kwargs['save'] is not None:
        with Timer('Saving model'):
            with open(kwargs['save'], 'wb') as save_file:
                torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    main()
