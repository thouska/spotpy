# -*- coding: utf-8 -*-
"""
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Optimization Tool (SPOTPY).

:author: Tobias Houska

This is the parent class of all algorithms, which can handle the database
structure during the sample.
"""

import numpy as np

from .base import database

try:
    import tables
except ImportError:
    print(
        "ImportError: Pytables is not correctly installed. Please also make sure you",
        "installed the hdf5 extension (https://www.hdfgroup.org/downloads/hdf5/)",
    )
    raise
import sys


class hdf5(database):
    """
    A database class to store the result in hdf5 tables.

    This is only available if PyTables is installed
    """

    def get_table_def(self):
        """
        Returns a dict of column definitions using multidimensional
        hdf5 columns. Columns for parameters and likelihoods are atomic and resemble
        the csv datawriter. If ``save_sim=True``, the simulation array is saved as an array value in
        a single multidimensional table cell

        cf.: https://www.pytables.org/usersguide/tutorials.html#multidimensional-table-cells-and-automatic-sanity-checks

        """
        # Position of likelihood columns
        like_pos = 0
        # Start position of parameter columns
        param_pos = np.array(self.like).size
        # Start position of simulation columns
        sim_pos = param_pos + np.array(self.randompar).size
        chain_pos = sim_pos

        dtype = np.dtype(self.db_precision)
        columns = {
            self.header[i]: tables.Col.from_dtype(dtype, pos=i)
            for i in range(like_pos, sim_pos)
        }

        if self.save_sim:
            # Get the shape of the simulation
            sim_shape = np.array(self.simulations).shape
            # Get the appropriate dtype for the n-d cell
            # (tables.Col.from_dtype does not take a shape parameter)
            sim_dtype = np.dtype((self.db_precision, sim_shape))
            columns["simulation"] = tables.Col.from_dtype(sim_dtype, pos=sim_pos)
            chain_pos += 1
        # Add a column chains
        columns["chains"] = tables.UInt16Col(pos=chain_pos)

        return columns

    def __init__(self, *args, **kwargs):
        """
        Create a new datawriter for hdf5 files
        :param args:
        :param kwargs:
        """
        # init base class
        super(hdf5, self).__init__(*args, **kwargs)
        # store init item only if dbinit
        if not kwargs.get("dbappend", False):
            # Create an open file, which needs to be closed after the sampling
            self.db = tables.open_file(self.dbname + ".h5", "w", self.dbname)
            self.table = self.db.create_table(
                "/", self.dbname, description=self.get_table_def()
            )
        else:
            # Continues writing file
            self.db = tables.open_file(self.dbname + ".h5", "a")
            self.table = self.db.root[self.dbname]

    def save(self, objectivefunction, parameterlist, simulations=None, chains=1):
        new_row = self.table.row

        coll = self.dim_dict["like"](objectivefunction) + self.dim_dict["par"](
            parameterlist
        )
        for header, value in zip(self.header, coll):
            new_row[header] = value
        if self.save_sim:
            new_row["simulation"] = simulations
        new_row["chains"] = chains
        new_row.append()

    def finalize(self):
        self.db.close()

    def getdata(self):
        with tables.open_file(self.dbname + ".h5", "a") as db:
            return db.root[self.dbname][:]
