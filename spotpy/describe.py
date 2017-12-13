# -*- coding: utf-8 -*-
"""
A collection of helper functions to describe spotpy setups

Usage:

>>> spotpy.describe.sampler(sampler)
>>> spotpy.describe.setup(model)

Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Philipp Kraft


"""
import sys

from .parameter import get_parameters_array
if sys.version_info.major>=3:
    from inspect import getdoc as _getdoc
    unicode = str
else:
    def _getdoc(obj):
        u = type(obj).__doc__.decode(encoding='utf-8', errors='ignore')
        return '\n'.join(l.strip() for l in u.split('\n') if l.strip())


def describe(obj):
    """
    Returns a long string description of a sampler with its model
    :param obj: A sampler
    :return: str
    """
    return u'Sampler:\n--------\n{}\n\nModel:\n------\n{}'.format(sampler(obj), setup(obj.setup))


def sampler(obj):
    """
    Returns a string representation of the sampler.
    By design, it is rather verbose and returns a
    large multiline description
    :return:
    """
    s = u'' + type(obj).__name__
    s += _getdoc(obj) + '\n'
    s += '\n    db format: ' + obj.dbformat
    s += '\n    db name: ' + obj.dbname
    s += '\n    save simulation: ' + str(obj.save_sim)
    s += '\n    parallel: ' + type(obj.repeat).__module__.split('.')[-1]
    return s


def setup(obj):
    """
    Describes a spotpy setup using its class name, docstring and parameters
    :param setup: A spotpy compatible model setup
    :return: A describing string
    """

    # Get class name
    s = unicode(type(obj).__name__)
    # Add hbar
    s += '\n' + 30 * '-' + '\n\n'

    # Add doc string
    mdoc = _getdoc(obj)

    s += mdoc + '\n'

    # Get parameters from class
    params = get_parameters_array(type(obj))

    # Get parameters from obj.parameters if it is not a function
    if hasattr(obj, 'parameter') and not callable(obj.parameters):
        try:
            params += list(obj.parameters)
        except TypeError:
            raise ValueError('obj.parameter must be either callable or list of parameters')

    params = '\n'.join(' - {p}'.format(p=p) for p in params)

    # Add parameters to string
    s += '\n'
    s += 'Parameters:\n{params}'.format(params=params)

    return s
