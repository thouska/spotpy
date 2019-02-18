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
import time
from itertools import product
import sys
if sys.version_info[0] >= 3:
    unicode = str

from importlib import import_module

class database(object):
    """
    Parent class for database. It can handle the basic functionalities of all
    databases.
    """

    def __init__(self, dbname, parnames, like, randompar, simulations=None,
                 chains=1, save_sim=True, db_precision=np.float32, **kwargs):
        # Just needed for the first line in the database
        self.chains = chains
        self.dbname = dbname
        self.like = like
        self.randompar = randompar
        self.simulations = simulations
        self.save_sim = save_sim
        self.db_precision = db_precision
        if not save_sim:
            simulations = None
        self.dim_dict = {}
        self.singular_data_lens = [self._check_dims(name, obj) for name, obj in [(
            'like', like), ('par', randompar), ('simulation', simulations)]]
        self._make_header(simulations, parnames)

        self.last_flush = time.time()

    def _check_dims(self, name, obj):
        '''checks dimensionality of obj and stores function in dict'''
        if obj is None:
            # None object
            self.dim_dict[name] = self._empty_list
            return (0,)
        elif hasattr(obj, '__len__'):
            if hasattr(obj, 'shape'):
                # np.array style obj
                self.dim_dict[name] = self._array_to_list
                return obj.shape
            elif all([hasattr(x, '__len__') for x in obj]):
                # nested list, checked only for singular nesting
                # assumes all lists have same length
                self.dim_dict[name] = self._nestediterable_to_list
                return (len(obj), len(obj[0]))
            else:
                # simple iterable
                self.dim_dict[name] = self._iterable_to_list
                return (len(obj),)
        else:
            # scalar (int, float)
            self.dim_dict[name] = self._scalar_to_list
            return (1,)

    def _empty_list(self, obj):
        return []

    def _scalar_to_list(self, obj):
        # print('scalar')
        return [obj]

    def _iterable_to_list(self, obj):
        # print('iterable')
        return list(obj)

    def _array_to_list(self, obj):
        # print('array')
        values = []
        for val in obj:
            values.append(val)
        return values
        # return obj.flatten().tolist()

    def _nestediterable_to_list(self, obj):
        # print('nested')
        values = []
        for nestedlist in obj:
            # print(len(nestedlist))
            for val in nestedlist:
                values.append(val)
        # print(len(values))
        return values
        # return np.array(obj).flatten().tolist()

    def _make_header(self, simulations, parnames):
        self.header = []
        self.header.extend(['like' + '_'.join(map(str, x))
                            for x in product(*self._tuple_2_xrange(self.singular_data_lens[0]))])
        self.header.extend(['par{0}'.format(x) for x in parnames])
        # print(self.singular_data_lens[2])
        # print(type(self.singular_data_lens[2]))
        if self.save_sim:
            for i in range(len(simulations)):
                if isinstance(simulations[0], list) or type(simulations[0]) == type(np.array([])):
                    for j in range(len(simulations[i])):
                        self.header.extend(['simulation' + str(i + 1) + '_' + str(j + 1)])
                else:
                    self.header.extend(['simulation' + '_' + str(i)])
                    # for x in product(*self._tuple_2_xrange(self.singular_data_lens[2]))])

        self.header.append('chain')

    def _tuple_2_xrange(self, t):
        return (range(1, x + 1) for x in t)


class noData(database):
    """
    This class saves the process in the working storage. It can be used if
    safety matters.
    """

    def __init__(self, *args, **kwargs):
        pass

    def save(self, objectivefunction, parameterlist, simulations=None, chains=1):
        pass

    def finalize(self):
        pass

    def getdata(self):
        pass

class custom(database):
    """
    This class is a simple wrapper over the database API, and can be used
    when the user provides a custom save function.
    """

    def __init__(self, *args, **kwargs):
        if 'setup' not in kwargs:
            raise ValueError("""
                You are using the 'custom' Datawriter. To use it, the
                setup must be specified on creation, but it is missing
            """)
        self.setup = kwargs['setup']
        if not hasattr(self.setup, 'save'):
            raise AttributeError('Your setup needs a "save" method in order to use the "custom" dbformat')

        super(custom, self).__init__(*args, **kwargs)

    def save(self, objectivefunction, parameterlist, simulations, *args, **kwargs):
        self.setup.save(objectivefunction, parameterlist, simulations, *args, **kwargs)

    def finalize(self):
        pass

    def getdata(self):
        pass



