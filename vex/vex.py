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
import collections
import enum

vex = ctypes.cdll.LoadLibrary('./vex.so')


class ImmutableDict(collections.Mapping):

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return repr(self._data)


def send_string(s):
    return s.encode('utf-8')


def get_string(s):
    return s.decode('utf-8')


def get_array(obj, count, get_element, value_type):
    return [value_type(get_element(obj, i)) for i in range(count(obj))]


def get_dict(obj, check, get_key, get_value, value_type, key_type=get_string):
    out_dict = {}
    while check(obj):
        out_dict[key_type(get_key())] = value_type(get_value())
    return ImmutableDict(out_dict)


def get_class_members_repr(obj):
    v = vars(object)
    s = ["{0}: {1}".format(k, v) for k, v in v.items()]
    return type(obj).__name__ + ", ".join(s)


vex.source_def_name.restype = ctypes.c_char_p
vex.source_names_get.restype = ctypes.c_char_p
vex.source_ra.restype = ctypes.c_double
vex.source_dec.restype = ctypes.c_double
vex.source_cal_code.restype = ctypes.c_char


class Source(object):
    MAX_SRCNAME_LENGTH = 12

    def __init__(self, obj):
        self.def_name = get_string(vex.source_def_name(obj))
        self.ra = vex.source_ra(obj)
        self.dec = vex.source_dec(obj)
        self.cal_code = vex.source_cal_code(obj)
        self.source_names = get_array(obj, vex.source_names_count, vex.source_names_get, get_string)

    def __repr__(self):
        return get_class_members_repr(self)


vex.interval_start.restype = ctypes.c_double
vex.interval_stop.restype = ctypes.c_double


class Interval(object):
    def __init__(self, obj):
        self.start = vex.interval_start(obj)
        self.stop = vex.interval_stop(obj)

    def __repr__(self):
        return get_class_members_repr(self)


vex.scan_def_name.restype = ctypes.c_char_p
vex.scan_intent.restype = ctypes.c_char_p
vex.scan_mode_def_name.restype = ctypes.c_char_p
vex.scan_source_def_name.restype = ctypes.c_char_p
vex.scan_size.restype = ctypes.c_double
vex.scan_mjd_vex.restype = ctypes.c_double
vex.scan_stations_get.restype = ctypes.c_bool
vex.scan_stations_ant.restype = ctypes.c_char_p
vex.scan_record_enable_get.restype = ctypes.c_bool
vex.scan_record_enable_ant.restype = ctypes.c_char_p
vex.scan_record_enable.restype = ctypes.c_bool


class Scan(Interval):
    def __init__(self, obj):
        super(Scan, self).__init__(obj)

        self.def_name = get_string(vex.scan_def_name(obj))
        self.intent = get_string(vex.scan_intent(obj))
        self.mode_def_name = get_string(vex.scan_mode_def_name(obj))
        self.source_def_name = get_string(vex.scan_source_def_name(obj))
        self.size = vex.scan_size(obj)
        self.mjd_vex = vex.scan_mjd_vex(obj)
        self.stations = get_dict(obj,
                                 check=vex.scan_stations_get,
                                 get_key=vex.scan_stations_ant,
                                 get_value=vex.scan_stations_interval,
                                 value_type=Interval)
        self.record_enable = get_dict(obj,
                                      check=vex.scan_record_enable_get,
                                      get_key=vex.scan_record_enable_ant,
                                      get_value=vex.scan_record_enable,
                                      value_type=bool)

    def __repr__(self):
        return get_class_members_repr(self)


vex.subband_freq.restype = ctypes.c_double
vex.subband_freq.restype = ctypes.c_double
vex.subband_side_band.restype = ctypes.c_char
vex.subband_pol.restype = ctypes.c_char


class Subband(object):
    def __init__(self, obj):
        self.freq = vex.subband_freq(obj)
        self.bandwidth = vex.subband_bandwidth(obj)
        self.side_band = vex.subband_side_band(obj)
        self.pol = vex.subband_pol(obj)

    def __repr__(self):
        return get_class_members_repr(self)


vex.if_name.restype = ctypes.c_char_p
vex.if_sslo.restype = ctypes.c_double
vex.if_side_band.restype = ctypes.c_char
vex.if_pol.restype = ctypes.c_char
vex.if_phase_cal_interval_mhz.restype = ctypes.c_float
vex.if_phase_cal_base_mhz.restype = ctypes.c_float
vex.if_comment.restype = ctypes.c_char_p
vex.if_band_name.restype = ctypes.c_char_p
vex.if_vlba_band_name.restype = ctypes.c_char_p
vex.if_lower_edge_freq.restype = ctypes.c_double


class IF(object):
    def __init__(self, obj):
        self.name = get_string(vex.if_name(obj))
        self.if_sslo = vex.if_sslo(obj)
        self.if_side_band = vex.if_side_band(obj)
        self.pol = vex.if_pol(obj)
        self.phase_cal_interval_mhz = vex.if_phase_cal_interval_mhz(obj)
        self.phase_cal_base_mhz = vex.if_phase_cal_base_mhz(obj)
        self.comment = get_string(vex.if_comment(obj))

        self.band_name = get_string(vex.if_band_name(obj))
        self.vlba_band_name = get_string(vex.if_vlba_band_name(obj))
        self.lower_edge_freq = vex.if_lower_edge_freq(obj)

    def __repr__(self):
        return get_class_members_repr(self)


vex.channel_if_name.restype = ctypes.c_char_p
vex.channel_bbc_freq.restype = ctypes.c_double
vex.channel_bbc_bandwidth.restype = ctypes.c_double
vex.channel_bbc_side_band.restype = ctypes.c_char
vex.channel_name.restype = ctypes.c_char_p
vex.channel_bbc_name.restype = ctypes.c_char_p
vex.channel_tone_get.restype = ctypes.c_uint


class Channel(object):
    def __init__(self, obj):
        self.record_chan = vex.channel_record(obj)
        self.subband_id = vex.channel_subband_id(obj)
        self.if_name = get_string(vex.channel_if_name(obj))
        self.bbc_freq = vex.channel_bbc_freq(obj)
        self.bbc_bandwidth = vex.channel_bbc_bandwidth(obj)
        self.bbc_side_band = vex.channel_bbc_side_band(obj)
        self.name = vex.channel_name(obj)
        self.bbc_name = get_string(vex.channel_bbc_name(obj))
        self.tones = get_array(obj,
                               count=vex.channel_tone_count,
                               get_element=vex.channel_tone_get,
                               value_type=int)
        self.thread_Id = vex.channel_thread_id(obj)

    def __repr__(self):
        return get_class_members_repr(self)


class DataFormat(enum.Enum):
    NONE = 0
    VDIF = 1
    LegacyVDIF = 2
    Mark5B = 3
    VLBA = 4
    VLBN = 5
    Mark4 = 6
    KVN5B = 7
    LBASTD = 8
    LBAVSOP = 9
    S2 = 10
    NumDataFormats = 11


class ToneSelection(enum.Enum):
    ToneSelectionVex = 0
    ToneSelectionNone = 1
    ToneSelectionEnds = 2
    ToneSelectionAll = 3
    ToneSelectionSmart = 4
    ToneSelectionMost = 5
    ToneSelectionUnknown = 6
    NumToneSelections = 7


class SamplingType(enum.Enum):
    SamplingReal = 0
    SamplingComplex = 1
    SamplingComplexDSB = 2
    NumSamplingTypes = 3


class DataSource(enum.Enum):
    DataSourceNone = 0
    DataSourceModule = 1
    DataSourceFile = 2
    DataSourceNetwork = 3
    DataSourceFake = 4
    DataSourceMark6 = 5
    DataSourceSharedMemory = 6
    DataSourceUnspecified = 7
    NumDataSources = 8


vex.stream_sample_rate.restype = ctypes.c_double
vex.stream_n_bit.restype = ctypes.c_uint
vex.stream_n_record_chan.restype = ctypes.c_uint
vex.stream_n_thread.restype = ctypes.c_uint
vex.stream_fanout.restype = ctypes.c_uint
vex.stream_v_dif_frame_size.restype = ctypes.c_uint
vex.stream_single_thread.restype = ctypes.c_bool
vex.stream_difx_tsys.restype = ctypes.c_double


class Stream(object):
    def __init__(self, obj):
        self.sample_rate = vex.stream_sample_rate(obj)
        self.n_bit = vex.stream_n_bit(obj)
        self.n_record_chan = vex.stream_n_record_chan(obj)
        self.n_thread = vex.stream_n_thread(obj)
        self.fanout = vex.stream_fanout(obj)
        self.v_dif_frame_size = vex.stream_v_dif_frame_size(obj)
        self.single_thread = vex.stream_single_thread(obj)
        self.threads = get_array(obj,
                                 count=vex.stream_thread_count,
                                 get_element=vex.stream_thread_get,
                                 value_type=int)
        self.data_format = DataFormat(vex.stream_format(obj))
        self.data_sampling = SamplingType(vex.stream_data_sampling(obj))
        self.data_source = DataSource(vex.stream_data_source(obj))
        self.difx_tsys = vex.stream_difx_tsys(obj)

    def __repr__(self):
        return get_class_members_repr(self)


vex.setup_ifs_get_name.restype = ctypes.c_char_p


class Setup(object):
    def __init__(self, obj):
        self.ifs = get_dict(obj,
                            check=vex.setup_ifs_get,
                            get_key=vex.setup_ifs_get_name,
                            get_value=vex.setup_ifs_get_if,
                            key_type=get_string,
                            value_type=IF)

        self.channels = get_array(obj,
                                  count=vex.setup_channel_count,
                                  get_element=vex.setup_channel_get,
                                  value_type=Channel)

        self.streams = get_array(obj,
                                 count=vex.setup_stream_count,
                                 get_element=vex.setup_stream_get,
                                 value_type=Stream)

    def __repr__(self):
        return get_class_members_repr(self)


vex.mode_def_name.restype = ctypes.c_char_p
vex.mode_setups_get_ant.restype = ctypes.c_char_p


class Mode(object):
    def __init__(self, obj):
        self.def_name = get_string(vex.mode_def_name(obj))
        self.subbands = get_array(obj,
                                  count=vex.mode_subband_count,
                                  get_element=vex.mode_subband_get,
                                  value_type=Subband)
        self.zoombands = get_array(obj,
                                   count=vex.mode_zoomband_count,
                                   get_element=vex.mode_zoomband_get,
                                   value_type=Subband)
        self.setups = get_dict(obj,
                               check=vex.mode_setups_get,
                               get_key=vex.mode_setups_get_ant,
                               get_value=vex.mode_setups_get_setup,
                               key_type=get_string,
                               value_type=Setup)

    def __repr__(self):
        return get_class_members_repr(self)


vex.load.argtypes = [ctypes.c_char_p]
vex.get_directory.restype = ctypes.c_char_p
vex.obs_start.restype = ctypes.c_double
vex.obs_stop.restype = ctypes.c_double


class Vex(object):
    def __init__(self, filename):
        obj = vex.load(send_string(filename))
        self.directory = get_string(vex.get_directory(obj))
        self.polarisations = vex.get_polarisations(obj)
        self.obs_start = float(vex.obs_start(obj))
        self.obs_stop = float(vex.obs_stop(obj))
        self.sources = get_array(obj,
                                 count=vex.source_count,
                                 get_element=vex.source_get,
                                 value_type=Source)
        self.scans = get_array(obj,
                               count=vex.vscan_count,
                               get_element=vex.vscan_get,
                               value_type=Scan)
        self.modes = get_array(obj,
                               count=vex.mode_count,
                               get_element=vex.mode_get,
                               value_type=Mode)


if __name__ == '__main__':
    vex_file = Vex('./v255ae.vex')

    print(vex_file.directory)
    print(vex_file.polarisations)
    print(vex_file.obs_start)
    print(vex_file.obs_stop)

    print(len(vex_file.sources))
    for source in vex_file.sources:
        print("   ", source)

    print(len(vex_file.scans))
    for scan in vex_file.scans:
        print("    ", scan)

    print(len(vex_file.modes))
    for mode in vex_file.modes:
        print("    ", mode)
