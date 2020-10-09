'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska, Benjamin Manns
'''

import unittest
import sys
import numpy as np

from spotpy.examples.tutorial_padds import padds_spot_setup
from tests.test_dds import FixedRandomizer

try:
    import spotpy
except ImportError:
    sys.path.append(".")
    import spotpy

import os
from spotpy.examples.spot_setup_dds import spot_setup
import json
from spotpy.algorithms.padds import chc, HVC

class TestPADDS(unittest.TestCase):
    def setUp(self):
        self.spot_setup = padds_spot_setup()
        self.rep = 1000
        self.timeout = 1  # Given in Seconds
        self.f_random = FixedRandomizer()

    def json_helper(self, run):
        with open(os.path.dirname(__file__) + "/padds_tests/run_" + str(run) + ".json") as f:
            data = json.load(f)
        return data

    def test_run_1(self):
        self.run_a_dds(1)

    def test_run_2(self):
        self.run_a_dds(2)

    def test_run_3(self):
        self.run_a_dds(3)

    def test_run_4(self):
        self.run_a_dds(4)

    def test_run_5(self):
        self.run_a_dds(5)

    def test_run_6(self):
        self.run_a_dds(6)

    def test_run_7(self):
        self.run_a_dds(7)

    def test_run_8(self):
        self.run_a_dds(8)

    def test_run_9(self):
        self.run_a_dds(9)

    def outside_bound(self, x_curr, min_bound, max_bound):
        out_left = min_bound > x_curr  # [x<x_curr self.min_bound, self.max_bound
        out_right = max_bound < x_curr

        self.assertNotIn(True, out_right)
        self.assertNotIn(True, out_left)

    def test_bounds(self):
        self.spot_setup = padds_spot_setup(False)
        sampler = spotpy.algorithms.padds(self.spot_setup, parallel="seq", dbname='test_DDS', dbformat="csv",
                                        sim_timeout=self.timeout)
        sampler._set_np_random(self.f_random)
        results = sampler.sample(100, 1)

        self.outside_bound(results[0]['sbest'], sampler.min_bound, sampler.max_bound)

    def test_chc(self):
        self.assertArrayEqual([0.01851852, 0. , 0.01851852, 0.01851852, 0.,0.01851852],
                              chc([[1,10], [2,9.8],[3,5] ,[4, 4], [8,2], [10,1]]))
        with open(os.path.dirname(__file__) + "/padds_tests/CHC_testdata.json") as f:
            data = json.load(f)
            for i, test_data in enumerate(data):
                self.assertArrayEqual(chc(np.array(test_data["p"])),test_data["res"],delta=1e-6)

    def test_hvc(self):
        with open(os.path.dirname(__file__) + "/padds_tests/HVC_testdata.json") as f:
            data = json.load(f)
            hvc = HVC(fakerandom=True)
            for i, test_data in enumerate(data):
                self.assertArrayEqual(hvc(np.array(test_data["p"])),test_data["res"],delta=1e-6)

    def assertArrayEqual(self, a, b, delta=None):
        for j, elem in enumerate(a):
            try:
                self.assertAlmostEqual(elem, b[j], delta=delta)
            except IndexError:
                self.assertRaises("Index out of bound for array b at index = "+str(j))

    def run_a_dds(self, run):
        original_result = self.json_helper(run)

        sampler = spotpy.algorithms.padds(self.spot_setup, parallel="seq", dbname='test_PADDS', dbformat="csv",
                                        sim_timeout=self.timeout, r=original_result["r_val"])
        sampler._set_np_random(self.f_random)


        if original_result.get("initial_objs") is not None:
            # if a parameter initialisation is given, test this:
            results = sampler.sample(original_result["evatrials"],
                                     original_result["trial_runs"], initial_objs=np.array(original_result["initial_objs"]), initial_params=np.array(original_result["initial_params"]), metric=original_result["metric"])
        else:
            results = sampler.sample(original_result["evatrials"],
                                     original_result["trial_runs"],metric=original_result["metric"])

        for t in range(original_result["trial_runs"]):
            print(results[t]["objfunc_val"])
            print(original_result["results"][t]["objfunc_val"])

            self.assertArrayEqual(original_result["results"][t]["objfunc_val"], results[t]["objfunc_val"], 1e-5)
            self.assertArrayEqual(original_result["results"][t]["sbest"], results[t]["sbest"], 1e-5)


if __name__ == '__main__':
    unittest.main()
