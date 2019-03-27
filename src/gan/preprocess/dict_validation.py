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


class ValidationError(RuntimeError):
    pass


def check_type(types, value):
    if len(types) == 0:
        return True
    elif None in types and value is None:
        return True
    else:
        return any(map(lambda t: isinstance(value, t), types))


def get_value(d, key, types=[], range_min=None, range_max=None, default_value=None):
    value = d.get(key, default_value)

    if not check_type(types, value):
        if len(types) == 1:
            raise ValidationError('{0} is not of type {1}'.format(key, types[0]))
        else:
            raise ValidationError('{0} must be one of the following types {1}'.format(key, types))

    if range_min is not None and value < range_min:
        raise ValidationError('{0} must be at least {1}'.format(key, range_min))
    if range_max is not None and value > range_max:
        raise ValidationError('{0} must be no greater than {1}'.format(key, range_max))

    return value
