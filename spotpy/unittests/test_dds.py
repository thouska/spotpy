import unittest
import sys

try:
    import spotpy
except ImportError:
    sys.path.append(".")
    import spotpy

import numpy as np
import os
from spotpy.examples.spot_setup_dds import spot_setup
from pprint import pprint
import json


class TestDDS(unittest.TestCase):
    def setUp(self):
        self.spot_setup = spot_setup()
        self.rep = 1000
        self.timeout = 1  # Given in Seconds


    def ackley10(self,vector):
        length = len(vector)
        sum1 = 0
        sum2 = 0
        for i in range(length):
            sum1 = sum1 + vector[i] ** 2
            sum2 = sum2 + np.cos(2 * np.pi * vector[i])
        return -20 * np.exp(-0.2 * (sum1 / length) ** 0.5) - np.exp(sum2 / length)


    def json_helper(self, run):
        with open(os.path.dirname(__file__)+"/DDS_references/run_"+str(run)+".json") as f:
            data = json.load(f)

        return data

    def func_switcher(self,name):
        if name == "ackley":
            return self.ackley10

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

    def run_a_dds(self,run):
        original_result = self.json_helper(run)
        sampler = spotpy.algorithms.DDS(self.spot_setup, parallel="seq", dbname='test_DDS', dbformat="csv",
                                        sim_timeout=self.timeout)


        results = sampler.sample(original_result["evatrials"],self.func_switcher(original_result['objfunc']),original_result["r_val"],original_result["trial_runs"])


        for t in range(original_result["trial_runs"]):
            #pprint(results)
            #pprint(original_result)
            # +self.assertEqual(203, len(results))
            print(results[t]["objfunc_val"],original_result["results"][t]["objfunc_val"])
            self.assertTrue(np.abs(results[t]["objfunc_val"]-original_result["results"][t]["objfunc_val"]) < 0.000001)
            py_sbest = results[t]["sbest"]
            matlb_sbest = original_result["results"][t]["sbest"]
            for k in range(len(py_sbest)):
                print(py_sbest[k],matlb_sbest[k])
                self.assertAlmostEqual(py_sbest[k],matlb_sbest[k],delta=0.00001)

            py_trial_initial = results[t]["trial_initial"]
            matlb_trial_initial = original_result["results"][t]["trial_initial"]
            for k in range(len(py_sbest)):
                print(py_trial_initial[k], matlb_trial_initial[k])
                self.assertTrue(np.abs(py_trial_initial[k] - matlb_trial_initial[k]) < 0.0001)




if __name__ == '__main__':
    unittest.main()
