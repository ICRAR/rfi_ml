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

import os
from os import makedirs
from os.path import exists
from timeit import default_timer

import h5py
import numpy as np
import pandas as pd
from astropy.utils.console import human_time
from torch.utils.data import Dataset

NUMBER_CHANNELS = 1
NUMBER_OF_CLASSES = 2


class H5Exception(Exception):
    pass


class RfiDataset(Dataset):
    def __init__(self, args, data_type, rank=None):
        self._h5_file = get_h5_file(args)
        self._group = self._h5_file[data_type]
        self._length = self._group.attrs['length_data']
        x_data = self._group['data']
        y_data = self._group['labels']
        if rank is None:
            self._x_data = np.copy(x_data)
            self._y_data = np.copy(y_data)
            print('Pid: {}\tType: {}\tRank: {}\tLength: {}'.format(os.getpid(), data_type, rank, self._length))
        else:
            section_length = self._length / args.num_processes
            start = rank * section_length
            if rank == args.num_processes - 1:
                self._x_data = x_data[start:]
                self._y_data = y_data[start:]
            else:
                self._x_data = x_data[start:start + section_length]
                self._y_data = y_data[start:start + section_length]
            self._length = len(y_data)
            print('Pid: {}\tRank: {}\tLength: {}\tStart: {}'.format(os.getpid(), rank, self._length, start))

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self._x_data[index], self._y_data[index]


def process_files(filename, rfi_label, sequence_length):
    """ Process a file and return the data and the labels """
    number_channels = NUMBER_CHANNELS
    files_to_process = []
    for ending in ['.txt', '_loc.txt']:
        complete_filename = filename + ending
        if os.path.exists(complete_filename):
            files_to_process.append(complete_filename)

    if len(files_to_process) != 2:
        print('The line counts do not match for: {0}'.format(filename))
        return

    # Load the files into numpy
    print('Loading: {0}'.format(files_to_process[0]))
    data_frame = pd.read_csv(files_to_process[0], header=None, delimiter=' ')
    data = data_frame.values.flatten()
    print('Loading: {0}'.format(files_to_process[1]))
    data_frame = pd.read_csv(files_to_process[1], header=None, delimiter=' ')
    labels = data_frame.values.flatten()

    # Check the lengths match
    length_data = len(data)
    assert length_data == len(labels), 'The line counts do not match for: {0}'.format(filename)

    # If substitute of the label is needed
    if rfi_label != 1:
        labels[labels == 1] = rfi_label

    numpy_data = np.zeros((length_data, number_channels, sequence_length))
    for i in reversed(range(sequence_length)):
        start = sequence_length - i - 1
        if start == 0:
            numpy_data[:, 0, i] = data
        else:
            numpy_data[start:, 0, i] = data[:-start]
    return numpy_data, labels


def build_data(args):
    """ Read data """
    output_file = os.path.join(args.data_path, args.data_file)
    if os.path.exists(output_file):
        with Timer('Checking HDF5 file'):
            with h5py.File(output_file, 'r') as h5_file:
                # Everything matches
                if h5_file.attrs['sequence_length'] == args.sequence_length and \
                                h5_file.attrs['validation_percentage'] == args.validation_percentage and \
                                h5_file.attrs['training_percentage'] == args.training_percentage:
                    # All good nothing to do
                    return

    # Open the output files
    with Timer('Processing input files'):
        data0, labels0 = process_files('../data/GMRT/impulsive_broadband_simulation_random_norfi', 0, args.sequence_length)
        data1, labels1 = process_files('../data/GMRT/impulsive_broadband_simulation_random_5p', 1, args.sequence_length)
        data2, labels2 = process_files('../data/GMRT/impulsive_broadband_simulation_random_10p', 1, args.sequence_length)
        data3, labels3 = process_files('../data/GMRT/repetitive_rfi_timeseries', 1, args.sequence_length)
        data4, labels4 = process_files('../data/GMRT/repetitive_rfi_random_timeseries', 1, args.sequence_length)

    # Concatenate
    with Timer('Concatenating data'):
        labels = np.concatenate((labels0, labels1, labels2, labels3, labels4))
        data = np.concatenate((data0, data1, data2, data3, data4))

    # Standardise and one hot
    with Timer('Standardise & One hot'):
        labels = one_hot(labels, NUMBER_OF_CLASSES)
        data = standardize(data)

    # Train/Validation/Test Split
    with Timer('Train/Validation/Test Split'):
        length_data = len(data)
        perm0 = np.arange(length_data)
        np.random.shuffle(perm0)
        data = data[perm0]
        labels = labels[perm0]
        split_point1 = int(length_data * args.training_percentage / 100.)
        split_point2 = int(length_data * (args.training_percentage + args.validation_percentage) / 100.)

        x_train = data[:split_point1]
        x_validation = data[split_point1:split_point2]
        x_test = data[split_point2:]
        y_train = labels[:split_point1]
        y_validation = labels[split_point1:split_point2]
        y_test = labels[split_point2:]

    with Timer('Saving to {0}'.format(output_file)):
        if not exists(args.data_path):
            makedirs(args.data_path)
        with h5py.File(output_file, 'w') as h5_file:
            h5_file.attrs['sequence_length'] = args.sequence_length
            h5_file.attrs['number_channels'] = NUMBER_CHANNELS
            h5_file.attrs['number_classes'] = NUMBER_OF_CLASSES
            h5_file.attrs['validation_percentage'] = args.validation_percentage
            h5_file.attrs['training_percentage'] = args.training_percentage
            h5_file.attrs['length_data'] = len(labels)

            training_group = h5_file.create_group('training')
            training_group.attrs['length_data'] = len(x_train)
            training_group.create_dataset('data', data=x_train, compression='gzip')
            training_group.create_dataset('labels', data=y_train, compression='gzip')

            validation_group = h5_file.create_group('validation')
            validation_group.attrs['length_data'] = len(x_validation)
            validation_group.create_dataset('data', data=x_validation, compression='gzip')
            validation_group.create_dataset('labels', data=y_validation, compression='gzip')

            test_group = h5_file.create_group('test')
            test_group.attrs['length_data'] = len(x_test)
            test_group.create_dataset('data', data=x_test, compression='gzip')
            test_group.create_dataset('labels', data=y_test, compression='gzip')


def get_h5_file(args):
    """ Read data """
    output_file = os.path.join(args.data_path, args.data_file)
    if os.path.exists(output_file):
        with Timer('Checking HDF5 file'):
            h5_file = h5py.File(output_file, 'r')
            # Everything matches
            if h5_file.attrs['sequence_length'] == args.sequence_length and \
                    h5_file.attrs['validation_percentage'] == args.validation_percentage and \
                    h5_file.attrs['training_percentage'] == args.training_percentage:
                return h5_file

    # The read data needs to be called first
    raise H5Exception('You need to call build data first')


def standardize(all_data):
    """ Standardize data """
    all_data = (all_data - np.mean(all_data)) / np.std(all_data)

    return all_data


def one_hot(labels, number_class):
    """ One-hot encoding """
    expansion = np.eye(number_class)
    y = expansion[:, labels].T
    assert y.shape[1] == number_class, "Wrong number of labels!"

    return y  # TODO: Look at using MultiLabelBinarizer


class Timer(object):
    def __init__(self, name=None, verbose=True):
        self.verbose = verbose
        self.name = '' if name is None else name
        self.timer = default_timer

    def __enter__(self):
        print('Pid: {}\t{}\tStarting timer'.format(os.getpid(), self.name))
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs
        if self.verbose:
            print('Pid: {}\t{}\tElapsed time: {}'.format(os.getpid(), self.name, human_time(self.elapsed)))
