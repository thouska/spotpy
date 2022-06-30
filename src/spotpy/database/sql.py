from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import sqlite3
import sys
from .base import database

if sys.version_info[0] >= 3:
    unicode = str


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
        self.db_cursor.execute('''CREATE TABLE IF NOT EXISTS  ''' + self.dbname + '''
                     (''' + ' real ,'.join(self.header) + ''')''')

    def save(self, objectivefunction, parameterlist, simulations=None, chains=1):
        coll = (self.dim_dict['like'](objectivefunction) +
                self.dim_dict['par'](parameterlist) +
                self.dim_dict['simulation'](simulations) +
                [chains])
        # Apply rounding of floats
        coll = map(self.db_precision, coll)
        self.db_cursor.execute(
            "INSERT INTO " + self.dbname + " VALUES (" + '"' + str('","'.join(map(str, coll))) + '"' + ")")

        self.db.commit()

    def finalize(self):
        self.db.close()

    def getdata(self):
        self.db = PickalableSQL3Connect(self.dbname + '.db')
        self.db_cursor = PickalableSQL3Cursor(self.db)

        if sys.version_info[0] >= 3:
            headers = [(row[1], "<f8") for row in
                       self.db_cursor.execute("PRAGMA table_info(" + self.dbname + ");")]
        else:
            # Workaround for python2
            headers = [(unicode(row[1]).encode("ascii"), unicode("<f8").encode("ascii")) for row in
                       self.db_cursor.execute("PRAGMA table_info(" + self.dbname + ");")]

        back = np.array([row for row in self.db_cursor.execute('SELECT * FROM ' + self.dbname)], dtype=headers)

        self.db.close()
        return back
