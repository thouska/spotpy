# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Optimization Tool (SPOTPY).

:author: Tobias Houska

This is the parent class of all algorithms, which can handle the database
structure during the sample.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from .base import database
import sys
if sys.version_info[0] >= 3:
    unicode = str


class ram(database):
    """
    This class saves the process in the working storage. It can be used if
    time matters.
    """

    def __init__(self, *args, **kwargs):
        # init base class
        super(ram, self).__init__(*args, **kwargs)
        # init the status vars
        self.ram = []

    def save(self, objectivefunction, parameterlist, simulations=None,
             chains=1):
        self.ram.append(tuple(self.dim_dict['like'](objectivefunction) +
                              self.dim_dict['par'](parameterlist) +
                              self.dim_dict['simulation'](simulations) +
                              [chains]))

    def finalize(self):
        """
        Is called in a last step of every algorithm.
        Forms the List of values into a strutured numpy array in order to have
        the same structure as a csv database.
        """
        dt = {'names': self.header,
              'formats': [np.float] * len(self.header)}
        i = 0
        Y = np.zeros(len(self.ram), dtype=dt)

        for line in self.ram:
            Y[i] = line
            i += 1

        self.data = Y

    def getdata(self):
        """
        Returns a finalized database"""
        return self.data
