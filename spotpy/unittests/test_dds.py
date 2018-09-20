import unittest
import sys



try:
    import spotpy
except ImportError:
    sys.path.append(".")
    import spotpy

from spotpy.tools import FixedRandomizer

import os
from spotpy.examples.spot_setup_dds import spot_setup
import json


class TestDDS(unittest.TestCase):
    def setUp(self):
        self.spot_setup = spot_setup()
        self.rep = 1000
        self.timeout = 1  # Given in Seconds
        self.f_random = FixedRandomizer()

    def json_helper(self, run):
        with open(os.path.dirname(__file__) + "/DDS_references/run_" + str(run) + ".json") as f:
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

    def run_a_dds(self, run):
        original_result = self.json_helper(run)

        self.spot_setup._objfunc_switcher(original_result['objfunc'])

        sampler = spotpy.algorithms.DDS(self.spot_setup, parallel="seq", dbname='test_DDS', dbformat="csv",
                                        sim_timeout=self.timeout)
        sampler._set_np_random(self.f_random)

        results = sampler.sample(original_result["evatrials"], original_result["r_val"], original_result["trial_runs"])

        for t in range(original_result["trial_runs"]):
            print(results[t]["objfunc_val"], original_result["results"][t]["objfunc_val"])
            self.assertAlmostEqual(results[t]["objfunc_val"] , original_result["results"][t]["objfunc_val"],delta=0.000001)
            py_sbest = results[t]["sbest"]
            matlb_sbest = original_result["results"][t]["sbest"]
            for k in range(len(py_sbest)):
                print(py_sbest[k], matlb_sbest[k])
                self.assertAlmostEqual(py_sbest[k], matlb_sbest[k], delta=0.00001)

            py_trial_initial = results[t]["trial_initial"]
            matlb_trial_initial = original_result["results"][t]["trial_initial"]
            for k in range(len(py_sbest)):
                print(py_trial_initial[k], matlb_trial_initial[k])
                self.assertAlmostEqual(py_trial_initial[k],matlb_trial_initial[k], delta=0.0001)


if __name__ == '__main__':
    unittest.main()
