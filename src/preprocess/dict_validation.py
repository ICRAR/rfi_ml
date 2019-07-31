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
Provides basic validation for values stored in kwargs
"""


class ValidationError(RuntimeError):
    pass


def check_type(types: list, value):
    """
    Check that the type of a value is one of a list of types.

    Parameters
    ----------
    types : list
        List of types to check.
    value
        The value to check against the types

    Returns
    -------
    True if the type of value matches at least one type in the types list, false if not.
    """
    if len(types) == 0:
        return True
    elif None in types and value is None:
        return True
    else:
        return any(map(lambda t: isinstance(value, t), types))


def get_value(d: dict, key: str, types: list = [], range_min=None, range_max=None, default_value=None):
    """
    Get the value of a key from a dictionary with constraints.

    Parameters
    ----------
    d : dict
        The dictionary to get the value from.
    key : str
        The key
    types : list
        Possible types for this value
    range_min : number
        Minimum allowable value
    range_max : number
        Maximum allowable value
    default_value
        Default value to return if the value is not found in the dict.

    Returns
    -------
    The value from the dict, or the default value if the value is not in the dict.

    Raises
    -------
    ValidationError
        - If the type of the value is incorrect.
        - If the value is out of range
    """
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
