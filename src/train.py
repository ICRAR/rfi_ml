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

import numpy as np
import torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from histogram import Histogram


def train(args, model, rfi_data, rank=0):
    # This is needed to "trick" numpy into using different seeds for different processes
    if args.seed is not None:
        np.random.seed(args.seed + rank)
    else:
        np.random.seed()

    train_loader = data.DataLoader(rfi_data.get_rfi_dataset('training', rank=rank), batch_size=args.batch_size, num_workers=1)
    test_loader = data.DataLoader(rfi_data.get_rfi_dataset('validation', rank=rank), batch_size=args.batch_size, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        # Adjust the learning rate
        adjust_learning_rate(optimizer, epoch, args.learning_rate_decay, args.start_learning_rate_decay, args.learning_rate)
        train_epoch(epoch, args, model, train_loader, optimizer)
        test_epoch(args, model, test_loader)


def train_epoch(epoch, args, model, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (x_data, periodogram_data, target) in enumerate(data_loader):
        x_data = Variable(x_data)
        periodogram_data = Variable(periodogram_data)
        target = Variable(target)
        optimizer.zero_grad()
        output = model(x_data, periodogram_data)
        loss = functional.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and batch_idx > 1:
            print('Pid: {}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid,
                epoch,
                batch_idx * len(x_data),
                len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.data[0])
            )


def build_histogram(output, target_column, histogram_data):
    for values, column in zip(output.data.numpy(), target_column.numpy()):
        histogram_data.append(values[column])


def test_epoch(args, model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    histogram_data = []
    for batch_index, (x_data, periodogram_data, target) in enumerate(data_loader):
        x_data = Variable(x_data, volatile=True)
        periodogram_data = Variable(periodogram_data, volatile=True)
        target = Variable(target)
        output = model(x_data, periodogram_data)
        test_loss += functional.binary_cross_entropy(output, target, size_average=False).data[0]     # sum up batch loss
        pred = output.data.max(1)[1]   # get the index of the max log-probability
        target_column = target.data.max(1)[1]
        correct += pred.eq(target_column).cpu().sum()
        build_histogram(output, target_column, histogram_data)
        if batch_index % args.log_interval == 0 and batch_index > 1:
            print('Pid: {}\tTest iteration: {}\tCorrect count: {}'.format(os.getpid(), batch_index, correct))

    test_loss /= len(data_loader.dataset)
    print('Pid: {}\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        os.getpid(),
        test_loss,
        correct,
        len(data_loader.dataset),
        100. * correct / len(data_loader.dataset))
    )
    histogram = Histogram(histogram_data, bins=10, number_range=(0.0, 1.0), histogram_type='numbers')
    print(histogram.horizontal())


def adjust_learning_rate(optimizer, epoch, learning_rate_decay, start_learning_rate_decay, learning_rate):
    """ Sets the learning rate to the initial LR decayed  """
    lr_decay = learning_rate_decay ** max(epoch + 1 - start_learning_rate_decay, 0.0)
    new_learning_rate = learning_rate * lr_decay
    print('Pid: {}\tNew learning rate: {}'.format(os.getpid(), new_learning_rate))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = new_learning_rate
