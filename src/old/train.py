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
import logging

import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from constants import NUMBER_OF_CLASSES
from old.histogram import Histogram

LOGGER = logging.getLogger(__name__)


def train(model, rfi_data, rank=0, **kwargs):
    # This is needed to "trick" numpy into using different seeds for different processes
    if kwargs['seed'] is not None:
        np.random.seed(kwargs['seed'] + rank)
    else:
        np.random.seed()

    train_loader = data.DataLoader(
        rfi_data.get_rfi_dataset('training', rank=rank, short_run_size=kwargs['short_run']),
        batch_size=kwargs['batch_size'],
        num_workers=1,
        pin_memory=kwargs['using_gpu'],
    )
    test_loader = data.DataLoader(
        rfi_data.get_rfi_dataset('validation', rank=rank, short_run_size=kwargs['short_run']),
        batch_size=kwargs['batch_size'],
        num_workers=1,
        pin_memory=kwargs['using_gpu'],
    )

    optimizer = optim.SGD(model.parameters(), lr=kwargs['learning_rate'], momentum=kwargs['momentum'])
    for epoch in range(1, kwargs['epochs'] + 1):
        # Adjust the learning rate
        adjust_learning_rate(optimizer, epoch, kwargs['learning_rate_decay'], kwargs['start_learning_rate_decay'], kwargs['learning_rate'])
        train_epoch(epoch, model, train_loader, optimizer, kwargs['log_interval'])
        test_epoch(model, test_loader, kwargs['log_interval'])


def train_epoch(epoch, model, data_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (x_data_raw, target) in enumerate(data_loader):
        # x_data_ts = Variable(x_data_ts)
        x_data_raw = Variable(x_data_raw)
        target = Variable(target)
        optimizer.zero_grad()
        output = model(x_data_raw)
        if type(output.data) == torch.cuda.DoubleTensor:
            output = output.cpu()
        loss = functional.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and batch_idx > 1:
            LOGGER.info('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
                epoch,
                batch_idx * len(x_data_raw),
                len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.data[0])
            )


def build_histogram(output, target_column, histogram_data):
    for values, column in zip(output.data.numpy(), target_column.numpy()):
        histogram_data['all'].append(values[column])
        histogram_data[column].append(values[column])


def test_epoch(model, data_loader, log_interval):
    model.eval()
    test_loss = 0
    correct = 0
    histogram_data = {key: [] for key in range(NUMBER_OF_CLASSES)}
    histogram_data['all'] = []
    for batch_index, (x_data_raw, target) in enumerate(data_loader):
        # x_data_ts = Variable(x_data_ts, volatile=True)
        x_data_raw = Variable(x_data_raw, volatile=True)
        target = Variable(target)
        output = model(x_data_raw)
        if type(output.data) == torch.cuda.DoubleTensor:
            output = output.cpu()
        test_loss += functional.binary_cross_entropy(output, target).data[0]
        pred = output.data.max(1)[1]
        target_column = target.data.max(1)[1]
        correct += pred.eq(target_column).sum()
        build_histogram(output, target_column, histogram_data)

        if batch_index % log_interval == 0 and batch_index > 1:
            LOGGER.info('Test iteration: {}, Correct count: {}'.format(batch_index, correct))

    test_loss /= len(data_loader.dataset)
    LOGGER.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(data_loader.dataset),
        100. * correct / len(data_loader.dataset))
    )
    for key, value in histogram_data.items():
        histogram = Histogram(
            value,
            title='Percentage of Correctly Predicted: {}'.format(key),
            bins=10,
            number_range=(0.0, 1.0),
            histogram_type='numbers'
        )
        print(histogram.horizontal())


def adjust_learning_rate(optimizer, epoch, learning_rate_decay, start_learning_rate_decay, learning_rate):
    """ Sets the learning rate to the initial LR decayed  """
    lr_decay = learning_rate_decay ** max(epoch + 1 - start_learning_rate_decay, 0.0)
    new_learning_rate = learning_rate * lr_decay
    LOGGER.info('New learning rate: {}'.format(new_learning_rate))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = new_learning_rate
