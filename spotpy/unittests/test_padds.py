import unittest
import sys
import numpy as np

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
        self.spot_setup = spot_setup()
        self.rep = 1000
        self.timeout = 1  # Given in Seconds
        self.f_random = FixedRandomizer()

    def json_helper(self, run):
        with open(os.path.dirname(__file__) + "/padds_tests/run_" + str(run) + ".json") as f:
            data = json.load(f)
        return data

    def test_run_1(self):
        self.run_a_dds(1)


    def run_a_dds(self, run):
        original_result = self.json_helper(run)

        self.spot_setup._objfunc_switcher(original_result['objfunc'])

        sampler = spotpy.algorithms.dds(self.spot_setup, parallel="seq", dbname='test_DDS', dbformat="csv",
                                        sim_timeout=self.timeout,r=original_result["r_val"])
        sampler._set_np_random(self.f_random)

        if original_result.get("s_initial") is not None:
            # if a parameter initialisation is given, test this:
            results = sampler.sample(original_result["evatrials"],
                                     original_result["trial_runs"], x_initial=original_result["s_initial"])
        else:
            results = sampler.sample(original_result["evatrials"],
                                     original_result["trial_runs"])

        for t in range(original_result["trial_runs"]):
            print(results[t]["objfunc_val"], -1*original_result["results"][t]["objfunc_val"])
            self.assertAlmostEqual(results[t]["objfunc_val"], -1*original_result["results"][t]["objfunc_val"],
                                   delta=0.000001)
            py_sbest = results[t]["sbest"]
            matlb_sbest = original_result["results"][t]["sbest"]
            for k in range(len(py_sbest)):
                print(py_sbest[k], matlb_sbest[k])
                self.assertAlmostEqual(py_sbest[k], matlb_sbest[k], delta=0.00001)

            py_trial_initial = results[t]["trial_initial"]
            matlb_trial_initial = original_result["results"][t]["trial_initial"]
            for k in range(len(py_sbest)):
                print(t, k, py_trial_initial[k], matlb_trial_initial[k])
                self.assertAlmostEqual(py_trial_initial[k], matlb_trial_initial[k], delta=0.0001)

    def test_own_initial_out_of_borders_ackley_1(self):
        self.spot_setup._objfunc_switcher("ackley")
        sampler = spotpy.algorithms.dds(self.spot_setup, parallel="seq", dbname='test_DDS', dbformat="csv",
                                        sim_timeout=self.timeout)
        self.assertRaises(ValueError,sampler.sample,1000, x_initial=np.random.uniform(-2, 2, 9) + [3])

    def test_own_initial_too_lees(self):
        self.spot_setup._objfunc_switcher("ackley")
        sampler = spotpy.algorithms.dds(self.spot_setup, parallel="seq", dbname='test_DDS', dbformat="csv",
                                        sim_timeout=self.timeout)
        self.assertRaises(ValueError, sampler.sample, 1000, x_initial=np.random.uniform(-2, 2, 9))

    def test_own_initial_too_much(self):
        self.spot_setup._objfunc_switcher("ackley")
        sampler = spotpy.algorithms.dds(self.spot_setup, parallel="seq", dbname='test_DDS', dbformat="csv",
                                        sim_timeout=self.timeout)
        self.assertRaises(ValueError, sampler.sample, 1000, x_initial=np.random.uniform(-2, 2, 11))


if __name__ == '__main__':
    unittest.main()
