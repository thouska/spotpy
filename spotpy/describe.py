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
from __future__ import division, absolute_import, unicode_literals
import sys
from .parameter import get_parameters_from_setup
from .algorithms._algorithm import _algorithm
if sys.version_info[0] >= 3:
    from inspect import getdoc as _getdoc
    unicode = str
else:
    def _getdoc(obj):
        u = obj.__doc__
        try:
            return u'\n'.join(l.strip() for l in u.split(u'\n') if l.strip())
        except UnicodeDecodeError:
            raise AssertionError(
                '{}: Docstring uses unicode but {} misses the line ``from __future__ import unicode_literals``'
                .format(obj, type(obj).__module__)
                )

try:
    from docutils.core import publish_string
except ImportError:
    publish_string = None


def describe(obj):
    """
    Returns a long string description of a sampler with its model
    :param obj: A sampler
    :return: str
    """
    return 'Sampler:\n--------\n{}\n\nModel:\n------\n{}'.format(sampler(obj), setup(obj.setup))


def sampler(obj):
    """
    Returns a string representation of the sampler.
    By design, it is rather verbose and returns a
    large multiline description
    :return:
    """
    cname = unicode(type(obj).__name__)
    s = [cname, '=' * len(cname), _getdoc(obj),
         '    db format: ' + obj.dbformat,
         '    db name: ' + obj.dbname,
         '    save simulation: ' + str(obj.save_sim),
         '    parallel: ' + type(obj.repeat).__module__.split('.')[-1]]
    return '\n'.join(s)


def setup(obj):
    """
    Describes a spotpy setup using its class name, docstring and parameters
    :param obj: A spotpy compatible model setup
    :return: A describing string
    """
    # Get class name
    cname = unicode(type(obj).__name__)
    # Add doc string
    mdoc = _getdoc(obj).strip('\n').replace('\r', '\n')
    # Get parameters from class
    params = '\n'.join(' - {p}'.format(p=unicode(p)) for p in get_parameters_from_setup(obj))
    parts = [cname, '=' * len(cname), mdoc, 'Parameters:', '-' * 11, params]
    return '\n'.join(parts)

