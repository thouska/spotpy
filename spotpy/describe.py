# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python (SPOTPY).
:author: Philipp Kraft

A collection of helper functions to describe spotpy setups

Usage:

>>> spotpy.describe.sampler(sampler)
>>> spotpy.describe.setup(model)
"""

import sys

from .parameter import get_parameters_from_setup
if sys.version_info[0] >= 3:
    from inspect import getdoc as _getdoc
    unicode = str
else:
    def _getdoc(obj):
        u = obj.__doc__.decode(encoding='utf-8', errors='ignore')
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
    s = unicode(type(obj).__name__)
    s += _getdoc(obj) + '\n'
    s += u'\n    db format: ' + obj.dbformat
    s += u'\n    db name: ' + obj.dbname
    s += u'\n    save simulation: ' + str(obj.save_sim)
    s += u'\n    parallel: ' + type(obj.repeat).__module__.split('.')[-1]
    return s


def setup(obj):
    """
    Describes a spotpy setup using its class name, docstring and parameters
    :param obj: A spotpy compatible model setup
    :return: A describing string
    """

    # Get class name
    s = unicode(type(obj).__name__)

    # Add doc string
    mdoc = _getdoc(obj)

    s += mdoc + '\n'

    # Get parameters from class

    params = '\n'.join(' - {p}'.format(p=p) for p in
                       get_parameters_from_setup(obj))

    # Add parameters to string
    s += '\n'
    s += 'Parameters:\n{params}'.format(params=params)

    return s
