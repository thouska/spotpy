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
from spotpy.examples.tutorial_padds import padds_spot_setup
from spotpy.describe import describe
import os

#https://docs.python.org/3/library/unittest.html

class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        # How many digits to match in case of floating point answers
        self.tolerance = 7
        #Create samplers for every algorithm:
        self.rep = 987
        self.timeout = 10 #Given in Seconds

        self.parallel = os.environ.get('SPOTPY_PARALLEL', 'seq')
        self.dbformat = "ram"

    def test_mc(self):
        sampler=spotpy.algorithms.mc(spot_setup(),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_lhs(self):
        sampler=spotpy.algorithms.lhs(spot_setup(),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_mle(self):
        sampler=spotpy.algorithms.mle(spot_setup(),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_mcmc(self):
        sampler=spotpy.algorithms.mcmc(spot_setup(used_algorithm='mcmc'),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_demcz(self):
        sampler=spotpy.algorithms.demcz(spot_setup(used_algorithm='demcz'),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep, convergenceCriteria=0)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_dream(self):
        sampler=spotpy.algorithms.dream(spot_setup(used_algorithm='dream'),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep, convergence_limit=0.9, runs_after_convergence=500)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)
        sampler.check_par_validity_bound(np.random.rand(10))
        sampler.check_par_validity_bound(sampler.status.params)
        sampler.check_par_validity_bound([-100,100,0])

    def test_sceua(self):
        sampler=spotpy.algorithms.sceua(spot_setup(used_algorithm='sceua'),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertLessEqual(len(results), self.rep) #Sceua save per definition not all sampled runs

        # We simply use a very low threshold to force a optimization search termination
        sampler = spotpy.algorithms.sceua(self.spot_setup, parallel=self.parallel, dbname='Rosen',
                                          dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep,ngs=10, kstop=3, peps=0.1, pcento=0.1)

    def test_abc(self):
        sampler=spotpy.algorithms.abc(spot_setup(used_algorithm='abc'),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_fscabc(self):
        sampler=spotpy.algorithms.fscabc(spot_setup(used_algorithm='fscabc'),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_rope(self):
        sampler=spotpy.algorithms.rope(spot_setup(),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_sa(self):
        sampler=spotpy.algorithms.sa(spot_setup(),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)
        
    def test_list(self):
        #generate a List sampler input
        sampler=spotpy.algorithms.mc(spot_setup(),parallel=self.parallel, dbname='Rosen', dbformat='csv', sim_timeout=self.timeout)
        sampler.sample(self.rep)

        sampler=spotpy.algorithms.list_sampler(spot_setup(),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_fast(self):
        sampler=spotpy.algorithms.fast(spot_setup(),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep, M=5)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep) #Si values should be returned

    def test_padds(self):
        sampler=spotpy.algorithms.padds(padds_spot_setup(),parallel=self.parallel, dbname='Rosen', dbformat=self.dbformat, sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results)+5, self.rep) #Si values should be returned

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove("Rosen.csv")
            os.remove("TestAlgorithms.csv")
        except (IOError, OSError):
            pass

if __name__ == '__main__':
    unittest.main(exit=False)
