# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska, Benjamin Manns
"""

import json
import os
import unittest

import numpy as np

import spotpy
from spotpy.examples.spot_setup_dds import spot_setup


# replaces numpy.random module in a way
class FixedRandomizerEndOfDataException(Exception):
    pass


class FixedRandomizer:
    def __init__(self):
        self.debug = False
        self.uniform_counter = 0
        self.normal_counter = 0
        self.uniform_list = list(
            np.loadtxt(os.path.dirname(__file__) + "/dds_tests/uniform_list.txt")
        )

        self.uniform_list *= 3
        self.max_normal_counter = 10000
        self.max_uniform_counter = 30000

        self.normal_list = list(
            np.loadtxt(os.path.dirname(__file__) + "/dds_tests/normal_list.txt")
        )

    def rand(self, dim_x=1, dim_y=1):
        x = dim_x * [0]
        for i in range(dim_x):
            if self.uniform_counter < self.max_uniform_counter:
                x[i] = self.uniform_list[self.uniform_counter]
                self.uniform_counter = self.uniform_counter + 1
                if self.debug:
                    print("fixrand::rand() counter = " + str(self.uniform_counter))
            else:
                raise FixedRandomizerEndOfDataException(
                    "No more data left. Counter is: " + str(self.uniform_counter)
                )
        if len(x) == 1:
            return x[0]
        else:
            return x

    def randint(self, x_from, x_to):
        vals = [j for j in range(x_from, x_to)]
        vals_size = len(vals)
        if vals_size == 0:
            raise ValueError("x_to >= x_from")
        fraq = 1.0 / vals_size
        if self.uniform_counter < self.max_uniform_counter:
            q_uni = self.uniform_list[self.uniform_counter]
            pos = int(np.floor(q_uni / fraq))
            self.uniform_counter += 1
            if self.debug:
                print("fixrand::randint() counter = " + str(self.uniform_counter))
            return vals[pos]
        else:
            raise FixedRandomizerEndOfDataException("No more data left.")

    def normal(self, loc, scale, size=1):
        x = []
        for j in range(size):
            if self.normal_counter < self.max_normal_counter:
                x.append(self.normal_list[self.normal_counter] * scale + loc)
                self.normal_counter += 1
                if self.debug:
                    print("fixrand::normal() counter = " + str(self.normal_counter))

            else:
                raise FixedRandomizerEndOfDataException("No more data left.")
        if len(x) == 1:
            return x[0]
        else:
            return x


class TestDDS(unittest.TestCase):
    def setUp(self):
        self.spot_setup = spot_setup()
        self.rep = 1000
        self.timeout = 1  # Given in Seconds
        self.f_random = FixedRandomizer()

    def json_helper(self, run):
        with open(
            os.path.dirname(__file__) + "/dds_tests/run_" + str(run) + ".json"
        ) as f:
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

    def not_working_run_7(self):
        self.run_a_dds(7)

    def test_run_own_initial_1(self):
        self.run_a_dds("own_input_1")

    def test_run_own_initial_2(self):
        self.run_a_dds("own_input_2")

    def outside_bound(self, x_curr, min_bound, max_bound):
        out_left = min_bound > x_curr  # [x<x_curr self.min_bound, self.max_bound
        out_right = max_bound < x_curr

        self.assertNotIn(True, out_right)
        self.assertNotIn(True, out_left)

    def test_bounds(self):
        self.spot_setup._objfunc_switcher("cmf_style")
        sampler = spotpy.algorithms.dds(
            self.spot_setup,
            parallel="seq",
            dbname="test_DDS",
            dbformat="csv",
            sim_timeout=self.timeout,
        )
        sampler._set_np_random(self.f_random)

        results = sampler.sample(1000, 1)
        print("results", results[0]["sbest"])
        print(sampler.min_bound, sampler.max_bound)
        self.outside_bound(results[0]["sbest"], sampler.min_bound, sampler.max_bound)

    def run_a_dds(self, run):
        original_result = self.json_helper(run)

        self.spot_setup._objfunc_switcher(original_result["objfunc"])

        sampler = spotpy.algorithms.dds(
            self.spot_setup,
            parallel="seq",
            dbname="test_DDS",
            dbformat="csv",
            sim_timeout=self.timeout,
            r=original_result["r_val"],
        )
        sampler._set_np_random(self.f_random)

        if original_result.get("s_initial") is not None:
            # if a parameter initialisation is given, test this:
            results = sampler.sample(
                original_result["evatrials"],
                original_result["trial_runs"],
                x_initial=original_result["s_initial"],
            )
        else:
            results = sampler.sample(
                original_result["evatrials"], original_result["trial_runs"]
            )

        for t in range(original_result["trial_runs"]):
            print(
                results[t]["objfunc_val"],
                -1 * original_result["results"][t]["objfunc_val"],
            )
            self.assertAlmostEqual(
                results[t]["objfunc_val"],
                -1 * original_result["results"][t]["objfunc_val"],
                delta=0.000001,
            )
            py_sbest = results[t]["sbest"]
            matlb_sbest = original_result["results"][t]["sbest"]
            for k in range(len(py_sbest)):
                print(py_sbest[k], matlb_sbest[k])
                self.assertAlmostEqual(py_sbest[k], matlb_sbest[k], delta=0.00001)

            py_trial_initial = results[t]["trial_initial"]
            matlb_trial_initial = original_result["results"][t]["trial_initial"]
            for k in range(len(py_sbest)):
                print(t, k, py_trial_initial[k], matlb_trial_initial[k])
                self.assertAlmostEqual(
                    py_trial_initial[k], matlb_trial_initial[k], delta=0.0001
                )

    def test_own_initial_out_of_borders_ackley_1(self):
        self.spot_setup._objfunc_switcher("ackley")
        sampler = spotpy.algorithms.dds(
            self.spot_setup,
            parallel="seq",
            dbname="test_DDS",
            dbformat="csv",
            sim_timeout=self.timeout,
        )
        self.assertRaises(
            ValueError,
            sampler.sample,
            1000,
            x_initial=np.random.uniform(-2, 2, 9) + [3],
        )

    def test_own_initial_too_lees(self):
        self.spot_setup._objfunc_switcher("ackley")
        sampler = spotpy.algorithms.dds(
            self.spot_setup,
            parallel="seq",
            dbname="test_DDS",
            dbformat="csv",
            sim_timeout=self.timeout,
        )
        self.assertRaises(
            ValueError, sampler.sample, 1000, x_initial=np.random.uniform(-2, 2, 9)
        )

    def test_own_initial_too_much(self):
        self.spot_setup._objfunc_switcher("ackley")
        sampler = spotpy.algorithms.dds(
            self.spot_setup,
            parallel="seq",
            dbname="test_DDS",
            dbformat="csv",
            sim_timeout=self.timeout,
        )
        self.assertRaises(
            ValueError, sampler.sample, 1000, x_initial=np.random.uniform(-2, 2, 11)
        )


if __name__ == "__main__":
    unittest.main()
