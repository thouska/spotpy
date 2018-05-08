# -*- coding: utf-8 -*-
'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska

This file holds the example code from the Rosenbrock tutorial web-documention.
'''

import unittest
try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy
import numpy as np
from spotpy.examples.spot_setup_rosenbrock import spot_setup
from spotpy.describe import describe

#https://docs.python.org/3/library/unittest.html

class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        # How many digits to match in case of floating point answers
        self.tolerance = 7
        #Create samplers for every algorithm:
        self.spot_setup = spot_setup()
        self.rep = 1000
        self.timeout = 10 #Given in Seconds

        self.parallel = "seq"
        self.dbformat = "csv"

    def test_mc(self):
        sampler=spotpy.algorithms.mc(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)

    def test_lhs(self):
        sampler=spotpy.algorithms.lhs(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)

    def test_mle(self):
        sampler=spotpy.algorithms.mle(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)

    def test_mcmc(self):
        sampler=spotpy.algorithms.mcmc(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)

    def test_demcz(self):
        sampler=spotpy.algorithms.demcz(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)

    def test_dream(self):
        sampler=spotpy.algorithms.dream(spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)

    def test_sceua(self):
        sampler=spotpy.algorithms.sceua(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)

    def test_abc(self):
        sampler=spotpy.algorithms.abc(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)

    def test_fscabc(self):
        sampler=spotpy.algorithms.fscabc(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)

    def test_rope(self):
        sampler=spotpy.algorithms.rope(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)

    def test_sa(self):
        sampler=spotpy.algorithms.sa(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)
        
    def test_list(self):
        sampler=spotpy.algorithms.list(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), self.rep, self.tolerance)

    def test_fast(self):
        sampler=spotpy.algorithms.fast(self.spot_setup,parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        results = sampler.sample(self.rep)
        self.assertAlmostEqual(len(results), 2, self.tolerance) #Si values should be returned

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove("Rosen.csv")

        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main(exit=False)
