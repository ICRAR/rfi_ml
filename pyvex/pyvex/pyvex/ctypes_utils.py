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

from .ImmutableDict import ImmutableDict


def send_string(s):
    return s.encode('utf-8')


def get_string(s):
    return s.decode('utf-8')


def get_array(obj, count, get_element, value_type):
    return [value_type(get_element(obj, i)) for i in range(count(obj))]


def get_dict(obj, check, get_key, get_value, value_type, key_type=get_string):
    out_dict = {}
    while True:
        more = check(obj)
        if more is False:
            break

        key = get_key()
        key = key_type(key)
        value = get_value()
        value = value_type(value)
        out_dict[key] = value
    return ImmutableDict(out_dict)


def get_class_members_repr(obj):
    v = vars(obj)
    s = ["{0}: {1}".format(k, v) for k, v in v.items()]
    return "{0} {1}".format(type(obj).__name__, ", ".join(s))