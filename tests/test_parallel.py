# -*- coding: utf-8 -*-
'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska
'''

import unittest
import spotpy
import numpy as np
from spotpy.examples.spot_setup_rosenbrock import spot_setup
from spotpy.describe import describe
import os

class TestParallel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # How many digits to match in case of floating point answers
        self.tolerance = 7
        #Create samplers for every algorithm:
        self.rep = 21
        self.timeout = 10 #Given in Seconds

        self.dbformat = "ram"

    def test_seq(self):
        sampler=spotpy.algorithms.mc(spot_setup(),parallel='seq', dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_mpc(self):
        sampler=spotpy.algorithms.mc(spot_setup(),parallel='mpc', dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_umpc(self):
        sampler=spotpy.algorithms.mc(spot_setup(),parallel='umpc', dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_mpi(self):
        """Forgiving test case for the MPI repeater object.

        Usage:
            mpirun -np N python -m unittest test_parallel.TestParallel.test_mpi
        """
        from warnings import warn

        try:
            from mpi4py import MPI
        except ImportError:
            warn('You must install mpi4py in order to run test_mpi()')
            return True

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size

        if size < 2:
            warn('You must run test_mpi() with at least two processes')
            return True

        sampler = spotpy.algorithms.mc(spot_setup(), parallel='mpi', dbname='Rosen', dbformat=self.dbformat,
                                       sim_timeout=self.timeout,
                                       parallel_kwargs=dict(mpicomm=comm))

        if rank == 0:
            # no special treatment for spotpy master
            sampler.sample(self.rep)
        else:
            # spotpy workers are terminated with an `exit()` call
            # so we have to catch this right here
            with self.assertRaises(SystemExit):
                sampler.sample(self.rep)

        if rank == 0:
            results = sampler.getdata()
            self.assertEqual(len(results), self.rep)


if __name__ == '__main__':
    unittest.main(exit=False)
