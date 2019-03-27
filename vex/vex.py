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

import ctypes
vex_lib = ctypes.cdll.LoadLibrary('./vex.so')

vex_lib.load.argtypes = [ctypes.c_char_p]
vex_lib.get_directory.restype = ctypes.c_char_p
vex_lib.obs_start.restype = ctypes.c_double
vex_lib.obs_stop.restype = ctypes.c_double

vex_lib.get_source_count.restype = ctypes.c_size_t
vex_lib.get_source_id_by_def_name.argtypes = [None, ctypes.c_char_p]
vex_lib.get_source.argtypes = [None, ctypes.c_uint]
vex_lib.get_source_by_def_name.argtypes = [None, ctypes.c_char_p]
vex_lib.get_source_by_source_name.argtypes = [None, ctypes.c_char_p]


def send_string(s):
    return s.encode('utf-8')


def get_string(s):
    return s.decode('utf-8')


class Source(object):

    MAX_SRCNAME_LENGTH = 12

    def __init__(self, source):
        self.obj = source

    @property
    def name(self):
        return get_string(vex_lib.source_name(self.obj))

    @property
    def has_name(self, name):
        return vex_lib.source_has_name(self.obj, send_string(name))

    def get_source_names(self):
        pass

    @property
    def ra(self):
        return vex_lib.source_ra(self.obj)

    @property
    def dec(self):
        return vex_lib.source_dec(self.obj)

    @property
    def cal_code(self):
        return vex_lib.source_cal_code(self.obj)


class Vex(object):
    def __init__(self, filename):
        self.obj = vex_lib.load(send_string(filename))

    def get_directory(self):
        return get_string(vex_lib.get_directory(self.obj))

    def get_polarisations(self):
        return vex_lib.get_polarisations(self.obj)

    def obs_start(self):
        return float(vex_lib.obs_start(self.obj))

    def obs_stop(self):
        return float(vex_lib.obs_stop(self.obj))

    def get_source_count(self):
        return vex_lib.get_source_count(self.obj)

    def get_source_id_by_def_name(self, name):
        return vex_lib.get_source_id_by_def_name(self.obj, send_string(name))

    def get_source(self, num):
        return vex_lib.get_source(self.obj, num)

    def get_source_by_def_name(self, name):
        return vex_lib.get_source_by_def_name(self.obj, send_string(name))

    def get_source_by_source_name(self, name):
        return vex_lib.get_source_by_source_name(self.obj, send_string(name))


if __name__ == '__main__':
    vex = Vex('./v255ae.vex')
    print(vex.get_directory())
    print(vex.get_polarisations())
    print(vex.obs_start())
    print(vex.obs_stop())
    print(vex.get_source_count())
