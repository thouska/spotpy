import unittest
import sys
import numpy as np

from spotpy.examples.tut_padds import padds_spot_setup
from spotpy.unittests.test_dds import FixedRandomizer

try:
    import spotpy
except ImportError:
    sys.path.append(".")
    import spotpy

import os
from spotpy.examples.spot_setup_dds import spot_setup
import json


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

    def assertArrayEqual(self, a, b, delta=None):
        for j, elem in enumerate(a):
            try:
                self.assertAlmostEqual(elem, b[j], delta=delta)
            except IndexError:
                self.assertRaises("Index out of bound for array b at index = "+str(j))

    def run_a_dds(self, run):
        original_result = self.json_helper(run)

        sampler = spotpy.algorithms.padds(self.spot_setup, parallel="seq", dbname='test_PADDS', dbformat="csv",
                                        sim_timeout=self.timeout,num_objs=2,r=original_result["r_val"])
        sampler._set_np_random(self.f_random)

        if original_result.get("s_initial") is not None:
            # if a parameter initialisation is given, test this:
            results = sampler.sample(original_result["evatrials"],
                                     original_result["trial_runs"], x_initial=original_result["s_initial"])
        else:
            results = sampler.sample(original_result["evatrials"],
                                     original_result["trial_runs"])

        for t in range(original_result["trial_runs"]):
            print(results[t]["objfunc_val"])
            print(original_result["results"][t]["objfunc_val"])

            self.assertArrayEqual(original_result["results"][t]["objfunc_val"], results[t]["objfunc_val"], 1e-5)
            self.assertArrayEqual(original_result["results"][t]["sbest"], results[t]["sbest"], 1e-5)


if __name__ == '__main__':
    unittest.main()
