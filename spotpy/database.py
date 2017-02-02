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
import io
from itertools import product


class database(object):
    """
    Parent class for database. It can handle the basic functionalities of all
    databases.
    """

    def __init__(self, dbname, parnames, like, randompar, simulations=None,
                 chains=1, save_sim=True, **kwargs):
        # Just needed for the first line in the database
        self.chains = chains
        self.dbname = dbname
        self.like = like
        self.randompar = randompar
        self.simulations = simulations
        if not save_sim:
            simulations = None
        self.dim_dict = {}
        self.singular_data_lens = [self._check_dims(name, obj) for name, obj in [(
            'like', like), ('par', randompar), ('simulation', simulations)]]
        self._make_header(parnames)

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
        return [obj]

    def _iterable_to_list(self, obj):
        return list(obj)

    def _array_to_list(self, obj):
        return obj.flatten().tolist()

    def _nestediterable_to_list(self, obj):
        return np.array(obj).flatten().tolist()

    def _make_header(self, parnames):
        self.header = []
        self.header.extend(['like' + '_'.join(map(str, x))
                            for x in product(*self._tuple_2_xrange(self.singular_data_lens[0]))])
        self.header.extend(['par{0}'.format(x) for x in parnames])
        self.header.extend(['simulation' + '_'.join(map(str, x))
                            for x in product(*self._tuple_2_xrange(self.singular_data_lens[2]))])
        self.header.append('chain')

    def _tuple_2_xrange(self, t):
        return (xrange(1, x + 1) for x in t)


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
        # store init item only if dbinit
        if kwargs.get('dbinit', True):
            self.save(self.like, self.randompar, self.simulations, self.chains)

    def save(self, objectivefunction, parameterlist, simulations=None,
             chains=1):
        self.ram.append(self.dim_dict['like'](objectivefunction) +
                        self.dim_dict['par'](parameterlist) +
                        self.dim_dict['simulation'](simulations) +
                        [chains])

    def finalize(self):
        dt = np.dtype({'names': self.header,
                       'formats': [np.float] * len(self.header)})

        # ignore the first initialization run to reduce the risk of different
        # objectivefunction mixing
        self.data = np.array(self.ram[1:]).astype(dt)

    def getdata(self):
        # Expects a finalized database
        return self.data


class csv(database):
    """
    This class saves the process in the working storage. It can be used if
    safety matters.
    """

    def __init__(self, *args, **kwargs):
        # init base class
        super(csv, self).__init__(*args, **kwargs)
        # Create a open file, which needs to be closed after the sampling
        self.db = io.open(self.dbname + '.csv', 'wb')
        # write header line
        self.db.write((','.join(self.header) + '\n').encode('utf-8'))
        # store init item only if dbinit
        if kwargs.get('dbinit', True):
            self.save(self.like, self.randompar, self.simulations, self.chains)

    def save(self, objectivefunction, parameterlist, simulations=None, chains=1):
        coll = (self.dim_dict['like'](objectivefunction) +
                self.dim_dict['par'](parameterlist) +
                self.dim_dict['simulation'](simulations) +
                [chains])
        try:
            # maybe apply a rounding for the floats?!
            self.db.write(
                (','.join(map(str, coll)) + '\n').encode('utf-8'))
        except IOError:
            input("Please close the file " + self.dbname +
                  " When done press Enter to continue...")
            self.db.write(
                (','.join(map(str, coll)) + '\n').encode('utf-8'))

    def finalize(self):
        self.db.close()

    def getdata(self, dbname=None):
        data = np.genfromtxt(
            self.dbname + '.csv', delimiter=',', names=True)[1:]
        return data
