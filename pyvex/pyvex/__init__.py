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
import enum
import os
from .ctypes_utils import get_string, send_string, get_array, get_dict, get_class_members_repr

path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, "..")
libs = [os.path.join(path, f) for f in os.listdir(path) if 'libpyvex' in f and ('so' in f or 'dll' in f or 'dynlib' in f)]

_vex = None
for lib in libs:
    try:
        _vex = ctypes.cdll.LoadLibrary(lib)
    except:
        pass

if _vex is None:
    raise ImportError("Could not import libpyvex from candidates: {0}".format(libs))


_vex.source_def_name.restype = ctypes.c_char_p
_vex.source_names_get.restype = ctypes.c_char_p
_vex.source_ra.restype = ctypes.c_double
_vex.source_dec.restype = ctypes.c_double
_vex.source_cal_code.restype = ctypes.c_char


class Source(object):
    MAX_SRCNAME_LENGTH = 12

    def __init__(self, obj):
        self.def_name = get_string(_vex.source_def_name(obj))
        self.ra = _vex.source_ra(obj)
        self.dec = _vex.source_dec(obj)
        self.cal_code = _vex.source_cal_code(obj)
        self.source_names = get_array(obj, _vex.source_names_count, _vex.source_names_get, get_string)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.interval_start.restype = ctypes.c_double
_vex.interval_stop.restype = ctypes.c_double


class Interval(object):
    def __init__(self, obj):
        self.start = _vex.interval_start(obj)
        self.stop = _vex.interval_stop(obj)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.scan_def_name.restype = ctypes.c_char_p
_vex.scan_intent.restype = ctypes.c_char_p
_vex.scan_mode_def_name.restype = ctypes.c_char_p
_vex.scan_source_def_name.restype = ctypes.c_char_p
_vex.scan_size.restype = ctypes.c_double
_vex.scan_mjd_vex.restype = ctypes.c_double
_vex.scan_stations_get.restype = ctypes.c_bool
_vex.scan_stations_ant.restype = ctypes.c_char_p
_vex.scan_record_enable_get.restype = ctypes.c_bool
_vex.scan_record_enable_ant.restype = ctypes.c_char_p
_vex.scan_record_enable.restype = ctypes.c_bool


class Scan(Interval):
    def __init__(self, obj):
        super(Scan, self).__init__(obj)

        self.def_name = get_string(_vex.scan_def_name(obj))
        self.intent = get_string(_vex.scan_intent(obj))
        self.mode_def_name = get_string(_vex.scan_mode_def_name(obj))
        self.source_def_name = get_string(_vex.scan_source_def_name(obj))
        self.size = _vex.scan_size(obj)
        self.mjd_vex = _vex.scan_mjd_vex(obj)
        self.stations = get_dict(obj,
                                 check=_vex.scan_stations_get,
                                 get_key=_vex.scan_stations_ant,
                                 get_value=_vex.scan_stations_interval,
                                 value_type=Interval)
        self.record_enable = get_dict(obj,
                                      check=_vex.scan_record_enable_get,
                                      get_key=_vex.scan_record_enable_ant,
                                      get_value=_vex.scan_record_enable,
                                      value_type=bool)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.subband_freq.restype = ctypes.c_double
_vex.subband_bandwidth.restype = ctypes.c_double
_vex.subband_side_band.restype = ctypes.c_char
_vex.subband_pol.restype = ctypes.c_char


class Subband(object):
    def __init__(self, obj):
        self.freq = _vex.subband_freq(obj)
        self.bandwidth = _vex.subband_bandwidth(obj)
        self.side_band = _vex.subband_side_band(obj)
        self.pol = _vex.subband_pol(obj)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.if_name.restype = ctypes.c_char_p
_vex.if_sslo.restype = ctypes.c_double
_vex.if_side_band.restype = ctypes.c_char
_vex.if_pol.restype = ctypes.c_char
_vex.if_phase_cal_interval_mhz.restype = ctypes.c_float
_vex.if_phase_cal_base_mhz.restype = ctypes.c_float
_vex.if_comment.restype = ctypes.c_char_p
_vex.if_band_name.restype = ctypes.c_char_p
_vex.if_vlba_band_name.restype = ctypes.c_char_p
_vex.if_lower_edge_freq.restype = ctypes.c_double


class IF(object):
    def __init__(self, obj):
        self.name = get_string(_vex.if_name(obj))
        self.if_sslo = _vex.if_sslo(obj)
        self.if_side_band = _vex.if_side_band(obj)
        self.pol = _vex.if_pol(obj)
        self.phase_cal_interval_mhz = _vex.if_phase_cal_interval_mhz(obj)
        self.phase_cal_base_mhz = _vex.if_phase_cal_base_mhz(obj)
        self.comment = get_string(_vex.if_comment(obj))

        self.band_name = get_string(_vex.if_band_name(obj))
        self.vlba_band_name = get_string(_vex.if_vlba_band_name(obj))
        self.lower_edge_freq = _vex.if_lower_edge_freq(obj)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.channel_if_name.restype = ctypes.c_char_p
_vex.channel_bbc_freq.restype = ctypes.c_double
_vex.channel_bbc_bandwidth.restype = ctypes.c_double
_vex.channel_bbc_side_band.restype = ctypes.c_char
_vex.channel_name.restype = ctypes.c_char_p
_vex.channel_bbc_name.restype = ctypes.c_char_p
_vex.channel_tone_get.restype = ctypes.c_uint


class Channel(object):
    def __init__(self, obj):
        self.record_chan = _vex.channel_record(obj)
        self.subband_id = _vex.channel_subband_id(obj)
        self.if_name = get_string(_vex.channel_if_name(obj))
        self.bbc_freq = _vex.channel_bbc_freq(obj)
        self.bbc_bandwidth = _vex.channel_bbc_bandwidth(obj)
        self.bbc_side_band = _vex.channel_bbc_side_band(obj)
        self.name = get_string(_vex.channel_name(obj))
        self.bbc_name = get_string(_vex.channel_bbc_name(obj))
        self.tones = get_array(obj,
                               count=_vex.channel_tone_count,
                               get_element=_vex.channel_tone_get,
                               value_type=int)
        self.thread_Id = _vex.channel_thread_id(obj)

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


_vex.stream_sample_rate.restype = ctypes.c_double
_vex.stream_n_bit.restype = ctypes.c_uint
_vex.stream_n_record_chan.restype = ctypes.c_uint
_vex.stream_n_thread.restype = ctypes.c_uint
_vex.stream_fanout.restype = ctypes.c_uint
_vex.stream_v_dif_frame_size.restype = ctypes.c_uint
_vex.stream_single_thread.restype = ctypes.c_bool
_vex.stream_difx_tsys.restype = ctypes.c_double


class Stream(object):
    def __init__(self, obj):
        self.sample_rate = _vex.stream_sample_rate(obj)
        self.n_bit = _vex.stream_n_bit(obj)
        self.n_record_chan = _vex.stream_n_record_chan(obj)
        self.n_thread = _vex.stream_n_thread(obj)
        self.fanout = _vex.stream_fanout(obj)
        self.v_dif_frame_size = _vex.stream_v_dif_frame_size(obj)
        self.single_thread = _vex.stream_single_thread(obj)
        self.threads = get_array(obj,
                                 count=_vex.stream_thread_count,
                                 get_element=_vex.stream_thread_get,
                                 value_type=int)
        self.data_format = DataFormat(_vex.stream_data_format(obj))
        self.data_sampling = SamplingType(_vex.stream_data_sampling(obj))
        self.data_source = DataSource(_vex.stream_data_source(obj))
        self.difx_tsys = _vex.stream_difx_tsys(obj)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.setup_ifs_get.restype = ctypes.c_bool
_vex.setup_ifs_get_name.restype = ctypes.c_char_p


class Setup(object):
    def __init__(self, obj):
        self.ifs = get_dict(obj,
                            check=_vex.setup_ifs_get,
                            get_key=_vex.setup_ifs_get_name,
                            get_value=_vex.setup_ifs_get_if,
                            key_type=get_string,
                            value_type=IF)

        self.channels = get_array(obj,
                                  count=_vex.setup_channel_count,
                                  get_element=_vex.setup_channel_get,
                                  value_type=Channel)

        self.streams = get_array(obj,
                                 count=_vex.setup_stream_count,
                                 get_element=_vex.setup_stream_get,
                                 value_type=Stream)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.mode_def_name.restype = ctypes.c_char_p
_vex.mode_setups_get_ant.restype = ctypes.c_char_p
_vex.mode_setups_get.restype = ctypes.c_bool


class Mode(object):
    def __init__(self, obj):
        self.def_name = get_string(_vex.mode_def_name(obj))
        self.subbands = get_array(obj,
                                  count=_vex.mode_subband_count,
                                  get_element=_vex.mode_subband_get,
                                  value_type=Subband)
        self.zoombands = get_array(obj,
                                   count=_vex.mode_zoomband_count,
                                   get_element=_vex.mode_zoomband_get,
                                   value_type=Subband)
        self.setups = get_dict(obj,
                               check=_vex.mode_setups_get,
                               get_key=_vex.mode_setups_get_ant,
                               get_value=_vex.mode_setups_get_setup,
                               key_type=get_string,
                               value_type=Setup)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.clock_mjd.restype = ctypes.c_double
_vex.clock_offset.restype = ctypes.c_double
_vex.clock_rate.restype = ctypes.c_double
_vex.clock_offset_epoch.restype = ctypes.c_double


class Clock(object):
    def __init__(self, obj):
        self.mjd_start = _vex.clock_mjd(obj)
        self.offset = _vex.clock_offset(obj)
        self.rate = _vex.clock_rate(obj)
        self.offset_epoch = _vex.clock_offset_epoch(obj)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.baseband_data_filename.restype = ctypes.c_char_p


class BasebandData(object):
    def __init__(self, obj):
        self.filename = get_string(_vex.baseband_data_filename(obj))
        self.recorder_id = _vex.baseband_data_recorer_id(obj)
        self.stream_id = _vex.baseband_data_stream_id(obj)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.network_data_port.restype = ctypes.c_char_p


class NetworkData(object):
    def __init__(self, obj):
        self.network_port = get_string(_vex.network_data_port(obj))
        self.window_size = _vex.network_data_window_size(obj)


_vex.antenna_name.restype = ctypes.c_char_p
_vex.antenna_def_name.restype = ctypes.c_char_p
_vex.antenna_x.restype = ctypes.c_double
_vex.antenna_y.restype = ctypes.c_double
_vex.antenna_z.restype = ctypes.c_double
_vex.antenna_dx.restype = ctypes.c_double
_vex.antenna_dy.restype = ctypes.c_double
_vex.antenna_dz.restype = ctypes.c_double
_vex.antenna_pos_epoch.restype = ctypes.c_double
_vex.antenna_axis_type.restype = ctypes.c_char_p
_vex.antenna_axis_offset.restype = ctypes.c_double
_vex.antenna_pol_convert.restype = ctypes.c_bool


class Antenna(object):
    def __init__(self, obj):
        self.name = get_string(_vex.antenna_name(obj))
        self.def_name = get_string(_vex.antenna_def_name(obj))
        self.x = _vex.antenna_x(obj)
        self.y = _vex.antenna_y(obj)
        self.z = _vex.antenna_z(obj)
        self.dx = _vex.antenna_dx(obj)
        self.dy = _vex.antenna_dy(obj)
        self.dz = _vex.antenna_dz(obj)
        self.pos_epoch = _vex.antenna_pos_epoch(obj)
        self.axis_type = get_string(_vex.antenna_axis_type(obj))
        self.axis_offset = _vex.antenna_axis_offset(obj)
        self.clocks = get_array(obj,
                                count=_vex.antenna_clock_count,
                                get_element=_vex.antenna_clock_get,
                                value_type=Clock)
        self.tcal_frequency = _vex.antenna_tcal_frequency(obj)
        self.pol_convert = _vex.antenna_pol_convert(obj)
        self.vsns = get_array(obj,
                              count=_vex.antenna_vsn_count,
                              get_element=_vex.antenna_vsn_get,
                              value_type=BasebandData)
        self.files = get_array(obj,
                               count=_vex.antenna_file_count,
                               get_element=_vex.antenna_file_get,
                               value_type=BasebandData)
        self.ports = get_array(obj,
                               count=_vex.antenna_port_count,
                               get_element=_vex.antenna_port_get,
                               value_type=NetworkData)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.eop_mjd.restype = ctypes.c_double
_vex.eop_tai_utc.restype = ctypes.c_double
_vex.eop_ut1_utc.restype = ctypes.c_double
_vex.eop_x_pole.restype = ctypes.c_double
_vex.eop_y_pole.restype = ctypes.c_double


class Eop(object):
    def __init__(self, obj):
        self.mjd = _vex.eop_mjd(obj)
        self.tai_utc = _vex.eop_tai_utc(obj)
        self.ut1_utc = _vex.eop_ut1_utc(obj)
        self.x_pole = _vex.eop_x_pole(obj)
        self.y_pole = _vex.eop_y_pole(obj)

    def __repr__(self):
        return get_class_members_repr(self)


_vex.exper_name.restype = ctypes.c_char_p


class Exper(Interval):
    def __init__(self, obj):
        super(Exper, self).__init__(obj)
        self.name = get_string(_vex.exper_name(obj))

    def __repr__(self):
        return get_class_members_repr(self)


_vex.load.argtypes = [ctypes.c_char_p]
_vex.get_directory.restype = ctypes.c_char_p


class Vex(object):
    def __init__(self, filename):
        obj = _vex.load(send_string(filename))
        if obj == 0:
            raise FileNotFoundError("Vex file {0} could not be found or loaded".format(filename))
        self.directory = get_string(_vex.get_directory(obj))
        self.polarisations = _vex.get_polarisations(obj)
        self.exper = Exper(_vex.get_exper(obj))
        self.sources = get_array(obj,
                                 count=_vex.source_count,
                                 get_element=_vex.source_get,
                                 value_type=Source)
        self.scans = get_array(obj,
                               count=_vex.vscan_count,
                               get_element=_vex.vscan_get,
                               value_type=Scan)
        self.modes = get_array(obj,
                               count=_vex.mode_count,
                               get_element=_vex.mode_get,
                               value_type=Mode)

        self.antennas = get_array(obj,
                                  count=_vex.antenna_count,
                                  get_element=_vex.antenna_get,
                                  value_type=Antenna)

        self.eops = get_array(obj,
                              count=_vex.eop_count,
                              get_element=_vex.eop_get,
                              value_type=Eop)
        _vex.free_vex(obj)
