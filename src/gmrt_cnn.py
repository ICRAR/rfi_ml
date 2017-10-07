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
from __future__ import print_function

import argparse

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.data as data

from train import RfiDataset, test_epoch, train
from utilities import NUMBER_CHANNELS, NUMBER_OF_CLASSES, Timer, build_data


class GmrtCNN(nn.Module):
    def __init__(self, keep_probability=0.5):
        super(GmrtCNN, self).__init__()
        self.keep_probability = keep_probability
        self.conv1 = nn.Conv1d(NUMBER_CHANNELS, 18, kernel_size=2, stride=1)
        self.conv1.double()     # Force the Conv1d to use a double
        self.max_pool1 = nn.MaxPool1d(2, stride=2)
        self.conv2 = nn.Conv1d(18, 36, kernel_size=2, stride=1)
        self.conv2.double()     # Force the Conv1d to use a double
        self.max_pool2 = nn.MaxPool1d(2, stride=2)
        self.conv3 = nn.Conv1d(36, 72, kernel_size=2, stride=1)
        self.conv3.double()     # Force the Conv1d to use a double
        self.max_pool3 = nn.MaxPool1d(2, stride=2)
        self.fc1 = nn.Linear(72, 36)
        self.fc1.double()       # Force the Conv1d to use a double
        self.fc2 = nn.Linear(36, NUMBER_OF_CLASSES)
        self.fc2.double()       # Force the Conv1d to use a double

    def forward(self, input_data):
        input_data = input_data
        x = self.max_pool1(functional.relu(self.conv1(input_data)))
        x = self.max_pool2(functional.relu(self.conv2(x)))
        x = self.max_pool3(functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = functional.dropout(x, p=self.keep_probability, training=self.training)
        x = self.fc1(x)
        x = self.fc2(x)
        x = functional.sigmoid(x)
        return x


def main():
    parser = argparse.ArgumentParser(description='GMRT CNN Training')
    parser.add_argument('--batch-size', type=int, default=5000, metavar='N', help='input batch size for training (default: 5000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--num-processes', type=int, default=4, metavar='N', help='how many training processes to use (default: 4)')
    parser.add_argument('--use_gpu', action='store_true', help='use the GPU if it is available', default=False)
    parser.add_argument('--data_path', default='./data', help='the path to the data file')
    parser.add_argument('--data_file', default='data.h5', help='the name of the data file')
    parser.add_argument('--sequence_length', type=int, default=16, help='how many elements in a sequence')
    parser.add_argument('--validation_percentage', type=int, default=10, help='amount of data used for validation')
    parser.add_argument('--training_percentage', type=int, default=80, help='amount of data used for training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--learning_rate_decay', type=float, default=0.8, metavar='LRD', help='the initial learning rate decay rate')
    parser.add_argument('--start_learning_rate_decay', type=int, default=2, help='the epoch to start applying the LRD')

    args = parser.parse_args()

    # Do this first so all the data is built before we go parallel and get race conditions
    with Timer('Reading data'):
        build_data(args)

    if torch.cuda.is_available() and args.use_gpu:
        # This uses the HOGWILD! approach to lock free SGD
        model = nn.DataParallel(GmrtCNN())

        # Train
        train(args, model)

    else:
        # This uses the HOGWILD! approach to lock free SGD
        model = GmrtCNN()
        model.share_memory()  # gradients are allocated lazily, so they are not shared here

        processes = []
        for rank in range(args.num_processes):
            p = mp.Process(target=train, args=(args, model, rank))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    with Timer('Reading data'):
        test_loader = data.DataLoader(RfiDataset(args, 'test'), batch_size=args.batch_size)

    print('Training all done')
    test_epoch(model, test_loader)


if __name__ == '__main__':
    main()
