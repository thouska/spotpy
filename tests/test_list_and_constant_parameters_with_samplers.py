"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska
"""

import inspect
import unittest
from itertools import cycle

import numpy as np

import spotpy
from spotpy import parameter
from spotpy.objectivefunctions import rmse


class Rosenbrock(object):
    """
    A 3 dimensional implementation of the Rosenbrock function

    Result at (1,1,1) is 0.
    """

    def simulation(self, vector):
        vector = np.array(vector)
        simulations = [
            sum(
                100.0 * (vector[1:] - vector[:-1] ** 2.0) ** 2.0
                + (1 - vector[:-1]) ** 2.0
            )
        ]
        return simulations

    def evaluation(self):
        observations = [0]
        return observations

    def objectivefunction(self, simulation, evaluation):
        objectivefunction = -rmse(evaluation=evaluation, simulation=simulation)
        return objectivefunction


class RosenbrockWithConstant(Rosenbrock):
    """
    A 3 dimensional implementation of the Rosenbrock function

    Result at (1,1,1) is 0.
    """

    c = parameter.Constant(0, doc="Constant offset, should stay 0")
    x = parameter.Uniform(
        -10, 10, 1.5, 3.0, -10, 10, doc="x value of Rosenbrock function"
    )
    y = parameter.Uniform(
        -10, 10, 1.5, 3.0, -10, 10, doc="y value of Rosenbrock function"
    )
    z = parameter.Uniform(
        -10, 10, 1.5, 3.0, -10, 10, doc="z value of Rosenbrock function"
    )


class RosenbrockWithList(Rosenbrock):
    """
    A 3 dimensional implementation of the Rosenbrock function

    Result at (1,1,1) is 0.
    """

    l = parameter.List(np.arange(0, 10), repeat=True, doc="list parameter for testing")
    x = parameter.Uniform(
        -10, 10, 1.5, 3.0, -10, 10, doc="x value of Rosenbrock function"
    )
    y = parameter.Uniform(
        -10, 10, 1.5, 3.0, -10, 10, doc="y value of Rosenbrock function"
    )
    z = parameter.Uniform(
        -10, 10, 1.5, 3.0, -10, 10, doc="z value of Rosenbrock function"
    )


def get_all_samplers():
    def use(cl):
        # Check if class name starts with an exclusion term
        return inspect.isclass(cl) and not any(
            [cl.__name__.startswith(ex) for ex in ("_", "list")]
        )

    return inspect.getmembers(spotpy.algorithms, use)


class TestConstantSetups(unittest.TestCase):
    def setUp(self):
        self.rep = 1000
        self.setup_constant = RosenbrockWithConstant
        self.setup_list = RosenbrockWithList

    def sampler_with_constant(self, sampler_class):

        sampler = sampler_class(self.setup_constant(), dbformat="ram", save_sim=False)
        sampler.sample(self.rep)

        self.assertTrue(
            all(line[1] == 0 for line in sampler.datawriter.ram),
            msg="Parameter c == 0 not true in all lines with sampler {}".format(
                sampler
            ),
        )

    def sampler_with_list(self, sampler_class, valid=False):
        sampler = sampler_class(self.setup_list(), dbformat="ram", save_sim=False)
        sampler.sample(self.rep)
        iterator = cycle(np.arange(0, 10))
        for i in range(self.rep):
            self.assertEqual(sampler.datawriter.ram[i][1], next(iterator))

    def test_abc_sampler(self):
        self.sampler_with_constant(spotpy.algorithms.abc)
        with self.assertRaises(TypeError):
            self.sampler_with_list(spotpy.algorithms.abc)

    def test_demcz_sampler(self):
        self.sampler_with_constant(spotpy.algorithms.demcz)
        with self.assertRaises(TypeError):
            self.sampler_with_list(spotpy.algorithms.demcz)

    def test_dream_sampler(self):
        self.sampler_with_constant(spotpy.algorithms.dream)
        with self.assertRaises(TypeError):
            self.sampler_with_list(spotpy.algorithms.dream)

    def test_fscabc_sampler(self):
        self.sampler_with_constant(spotpy.algorithms.fscabc)
        with self.assertRaises(TypeError):
            self.sampler_with_list(spotpy.algorithms.fscabc)

    def test_lhs_sampler(self):
        self.sampler_with_constant(spotpy.algorithms.lhs)
        with self.assertRaises(TypeError):
            self.sampler_with_list(spotpy.algorithms.lhs)

    def test_mc_sampler(self):
        self.sampler_with_constant(spotpy.algorithms.mc)
        self.sampler_with_list(spotpy.algorithms.mc, valid=True)

    def test_mcmc_sampler(self):
        self.sampler_with_constant(spotpy.algorithms.mcmc)
        with self.assertRaises(TypeError):
            self.sampler_with_list(spotpy.algorithms.mcmc)

    def test_mle_sampler(self):
        self.sampler_with_constant(spotpy.algorithms.mle)
        with self.assertRaises(TypeError):
            self.sampler_with_list(spotpy.algorithms.mle)

    def test_rope_sampler(self):
        self.sampler_with_constant(spotpy.algorithms.rope)
        with self.assertRaises(TypeError):
            self.sampler_with_list(spotpy.algorithms.rope)

    def test_sa_sampler(self):
        self.sampler_with_constant(spotpy.algorithms.sa)
        with self.assertRaises(TypeError):
            self.sampler_with_list(spotpy.algorithms.sa)

    def test_sceua_sampler(self):
        self.sampler_with_constant(spotpy.algorithms.sceua)
        with self.assertRaises(TypeError):
            self.sampler_with_list(spotpy.algorithms.sceua)


if __name__ == "__main__":
    unittest.main()
