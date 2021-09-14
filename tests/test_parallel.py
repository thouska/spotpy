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

        # Set number of repititions (not to small)
        self.rep = 200
        self.timeout = 10 #Given in Seconds
        self.dbformat = "ram"

    def test_seq(self):
        print('spotpy', spotpy.__version__, 'parallel=seq')
        sampler=spotpy.algorithms.mc(spot_setup(),parallel='seq', dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_mpc(self):
        print('spotpy', spotpy.__version__, 'parallel=mpc')
        sampler = spotpy.algorithms.mc(spot_setup(),parallel='mpc', dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_umpc(self):
        print('spotpy', spotpy.__version__, 'parallel=umpc')
        sampler=spotpy.algorithms.mc(spot_setup(),parallel='umpc', dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)


if __name__ == '__main__':
    unittest.main(exit=False)
