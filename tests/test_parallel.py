# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska
"""

import unittest

import spotpy
from spotpy.examples.spot_setup_rosenbrock import spot_setup


class TestParallel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # How many digits to match in case of floating point answers
        self.tolerance = 7
        # Create samplers for every algorithm:
        self.rep = 21
        self.timeout = 10  # Given in Seconds

        self.dbformat = "ram"

    def test_seq(self):
        sampler = spotpy.algorithms.mc(
            spot_setup(),
            parallel="seq",
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_mpc(self):
        sampler = spotpy.algorithms.mc(
            spot_setup(),
            parallel="mpc",
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)

    def test_umpc(self):
        sampler = spotpy.algorithms.mc(
            spot_setup(),
            parallel="umpc",
            dbname="Rosen",
            dbformat=self.dbformat,
            sim_timeout=self.timeout,
        )
        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(len(results), self.rep)


if __name__ == "__main__":
    unittest.main(exit=False)
