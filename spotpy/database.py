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
import time
from itertools import product
import sqlite3
import sys
if sys.version_info[0] >= 3:
    unicode = str


class database(object):
    """
    Parent class for database. It can handle the basic functionalities of all
    databases.
    """

    def __init__(self, dbname, parnames, like, randompar, simulations=None,
                 chains=1, save_sim=True, db_precision=np.float16, **kwargs):
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
        self._make_header(simulations,parnames)
        
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
        #print('scalar')
        return [obj]

    def _iterable_to_list(self, obj):
        #print('iterable')
        return list(obj)

    def _array_to_list(self, obj):
        #print('array')
        values = []        
        for val in obj:
            values.append(val)
        return values
        #return obj.flatten().tolist()

    def _nestediterable_to_list(self, obj):
        #print('nested')
        values = []        
        for nestedlist in obj:
            #print(len(nestedlist))
            for val in nestedlist:
                values.append(val)
        #print(len(values))
        return values
        #return np.array(obj).flatten().tolist()

    def _make_header(self, simulations,parnames):
        self.header = []
        self.header.extend(['like' + '_'.join(map(str, x))
                            for x in product(*self._tuple_2_xrange(self.singular_data_lens[0]))])
        self.header.extend(['par{0}'.format(x) for x in parnames])
        #print(self.singular_data_lens[2])
        #print(type(self.singular_data_lens[2]))        
        if self.save_sim:
            for i in range(len(simulations)):
                if type(simulations[0]) == type([]) or type(simulations[0]) == type(np.array([])):
                    for j in range(len(simulations[i])):
                        self.header.extend(['simulation' + str(i+1)+'_'+str(j+1)])
                else:
                    self.header.extend(['simulation' + '_'+str(i)])
                                    #for x in product(*self._tuple_2_xrange(self.singular_data_lens[2]))])

        self.header.append('chain')

    def _tuple_2_xrange(self, t):
        return (range(1, x + 1) for x in t)


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
            i+=1
        
        self.data = Y

    def getdata(self):
        """
        Returns a finalized database"""
        return self.data


class csv(database):
    """
    This class saves the process in the working storage. It can be used if
    safety matters.
    """

    def __init__(self, *args, **kwargs):
        # init base class
        super(csv, self).__init__(*args, **kwargs)
        # store init item only if dbinit
        if kwargs.get('dbinit', True):
            # Create a open file, which needs to be closed after the sampling
            self.db = io.open(self.dbname + '.csv', 'w')
            # write header line
            self.db.write(unicode(','.join(self.header) + '\n'))
        else:
            # Continues writing file
            self.db = io.open(self.dbname + '.csv', 'a')

    def save(self, objectivefunction, parameterlist, simulations=None, chains=1):
        coll = (self.dim_dict['like'](objectivefunction) +
                self.dim_dict['par'](parameterlist) +
                self.dim_dict['simulation'](simulations) +
                [chains])
        try:
            # Apply rounding of floats
            coll = map(self.db_precision, coll)
            self.db.write(
                ','.join(map(str, coll)) + '\n')
        except IOError:
            input("Please close the file " + self.dbname +
                  " When done press Enter to continue...")
            # Apply rounding of floats
            coll = map(self.db_precision, coll)
            self.db.write(
                ','.join(map(str, coll)) + '\n')
            
        acttime = time.time()
        # Force writing to disc at least every two seconds
        if acttime - self.last_flush >= 2:
            self.db.flush()
            self.last_flush = time.time()

    def finalize(self):
        self.db.close()

    def getdata(self):
        data = np.genfromtxt(self.dbname + '.csv', delimiter=',', names=True)
        return data


class PickalableSWIG:
    def __setstate__(self, state):
        self.__init__(*state['args'])
    def __getstate__(self):
        return {'args': self.args}


class PickalableSQL3Connect(sqlite3.Connection, PickalableSWIG):
    def __init__(self, *args,**kwargs):
        self.args = args
        sqlite3.Connection.__init__(self,*args,**kwargs)


class PickalableSQL3Cursor(sqlite3.Cursor, PickalableSWIG):
    def __init__(self, *args,**kwargs):
        self.args = args
        sqlite3.Cursor.__init__(self,*args,**kwargs)




class sql(database):

    """
    This class saves the process in the working storage. It can be used if
    safety matters.
    """

    def __init__(self, *args, **kwargs):
        import os
        # init base class
        super(sql, self).__init__(*args, **kwargs)
        # Create a open file, which needs to be closed after the sampling
        try:        
            os.remove(self.dbname + '.db')
        except:
            pass

        self.db = PickalableSQL3Connect(self.dbname + '.db')
        self.db_cursor = PickalableSQL3Cursor(self.db)
        # Create Table
#        self.db_cursor.execute('''CREATE TABLE IF NOT EXISTS  '''+self.dbname+'''
#                     (like1 real, parx real, pary real, simulation1 real, chain int)''')
        self.db_cursor.execute('''CREATE TABLE IF NOT EXISTS  '''+self.dbname+'''
                     ('''+' real ,'.join(self.header)+''')''')

    def save(self, objectivefunction, parameterlist, simulations=None, chains=1):
        coll = (self.dim_dict['like'](objectivefunction) +
                self.dim_dict['par'](parameterlist) +
                self.dim_dict['simulation'](simulations) +
                [chains])
        # Apply rounding of floats
        coll = map(self.db_precision, coll)
        try:
            self.db_cursor.execute("INSERT INTO "+self.dbname+" VALUES ("+'"'+str('","'.join(map(str, coll)))+'"'+")")

        except Exception:
            input("Please close the file " + self.dbname +
                  " When done press Enter to continue...")
            self.db_cursor.execute("INSERT INTO "+self.dbname+" VALUES ("+'"'+str('","'.join(map(str, coll)))+'"'+")")

        self.db.commit()

    def finalize(self):
        self.db.close()

    def getdata(self):
        self.db = PickalableSQL3Connect(self.dbname + '.db')
        self.db_cursor = PickalableSQL3Cursor(self.db)

        if sys.version_info[0] >= 3:
            headers = [(row[1],"<f8") for row in
                       self.db_cursor.execute("PRAGMA table_info(" + self.dbname+");")]
        else:
            # Workaround for python2
            headers = [(unicode(row[1]).encode("ascii"), unicode("<f8").encode("ascii")) for row in
                       self.db_cursor.execute("PRAGMA table_info(" + self.dbname + ");")]
        
        back = np.array([row for row in self.db_cursor.execute('SELECT * FROM ' + self.dbname)],dtype=headers)

        self.db.close()
        return back
        
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
                You must use either of ram, csv, sql or noData for your dbformat,
                OR implement a `save` function in your spot_setup class.
            """)
        self.setup = kwargs['setup']
        super(custom, self).__init__(*args, **kwargs)

    def save(self, objectivefunction, parameterlist, simulations, *args, **kwargs):
        self.setup.save(objectivefunction, parameterlist, simulations, *args, **kwargs)

    def finalize(self):
        pass

    def getdata(self):
        pass


def get_datawriter(dbformat, *args, **kwargs):
    """Given a dbformat (ram, csv, sql, noData, etc), return the constructor
        of the appropriate class from this file.
    """
    datawriter = globals()[dbformat](*args, **kwargs)
    return datawriter
