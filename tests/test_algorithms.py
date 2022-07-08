# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska
"""

import os
import unittest

import numpy as np

import spotpy
from spotpy.describe import describe
from spotpy.examples.spot_setup_hymod_python import spot_setup as spot_setup_hymod
from spotpy.examples.spot_setup_rosenbrock import spot_setup
from spotpy.likelihoods import gaussianLikelihoodMeasErrorOut as GausianLike

# https://docs.python.org/3/library/unittest.html


class TestAlgorithms(unittest.TestCase):
    def multi_obj_func(self, evaluation, simulation, params=None):
        # used to overwrite objective function in hymod example
        like1 = abs(spotpy.objectivefunctions.bias(evaluation, simulation))
        like2 = spotpy.objectivefunctions.rmse(evaluation, simulation)
        like3 = spotpy.objectivefunctions.rsquared(evaluation, simulation) * -1
        return np.array([like1, like2, like3])

    def setUp(self):
        # How many digits to match in case of floating point answers
        self.tolerance = 7
        # Create samplers for every algorithm:
        self.rep = 987
        self.timeout = 10  # Given in Seconds

        self.parallel = os.environ.get("SPOTPY_PARALLEL", "seq")
        self.dbformat = "ram"

    def test_mc(self):
        sampler = spotpy.algorithms.mc(
            spot_setup(),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_lhs(self):
        sampler = spotpy.algorithms.lhs(
            spot_setup(),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_mle(self):
        sampler = spotpy.algorithms.mle(
            spot_setup(),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_mcmc(self):
        sampler = spotpy.algorithms.mcmc(
            spot_setup(GausianLike),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_demcz(self):
        sampler = spotpy.algorithms.demcz(
            spot_setup(GausianLike),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep, convergenceCriteria=0)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_dream(self):
        sampler = spotpy.algorithms.dream(
            spot_setup(GausianLike),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep, convergence_limit=0.9, runs_after_convergence=500)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_sceua(self):
        sampler = spotpy.algorithms.sceua(
            spot_setup(),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertLessEqual(
            len(results), self.rep
        )  # Sceua save per definition not all sampled runs

    def test_abc(self):
        sampler = spotpy.algorithms.abc(
            spot_setup(),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_fscabc(self):
        sampler = spotpy.algorithms.fscabc(
            spot_setup(),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_rope(self):
        sampler = spotpy.algorithms.rope(
            spot_setup(),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_sa(self):
        sampler = spotpy.algorithms.sa(
            spot_setup(),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_list(self):
        # generate a List sampler input
        sampler = spotpy.algorithms.mc(
            spot_setup(),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat="csv",
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)

        sampler = spotpy.algorithms.list_sampler(
            spot_setup(),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_fast(self):
        sampler = spotpy.algorithms.fast(
            spot_setup(),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep, M=5)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)  # Si values should be returned

    def test_padds(self):
        sampler = spotpy.algorithms.padds(
            spot_setup_hymod(self.multi_obj_func),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(int(self.rep * 0.5), metric="ones")
        results = sampler.getdata()
        self.assertEqual(len(results) + 5, int(self.rep * 0.5))

    def test_nsgaii(self):
        generations = 20
        n_pop = 10
        sampler = spotpy.algorithms.NSGAII(
            spot_setup_hymod(self.multi_obj_func),
            parallel=self.parallel,
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(generations, n_obj=3, n_pop=n_pop)
        results = sampler.getdata()
        self.assertLessEqual(len(results), generations * n_pop)

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove("Rosen.csv")

        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main(exit=False)
