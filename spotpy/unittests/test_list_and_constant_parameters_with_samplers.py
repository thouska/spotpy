import unittest
import numpy as np
import inspect
try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy

from spotpy import parameter
from spotpy.objectivefunctions import rmse


class Rosenbrock(object):
    """
    A 3 dimensional implementation of the Rosenbrock function

    Result at (1,1,1) is 0.
    """

    def simulation(self, vector):
        x = np.array(vector)
        simulations = [sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)]
        return simulations

    def evaluation(self):
        observations = [0]
        return observations

    def objectivefunction(self, simulation, evaluation):
        objectivefunction = -rmse(evaluation=evaluation, simulation=simulation)
        return objectivefunction


class RosenbrockWithConstantAndList(Rosenbrock):
    """
    A 3 dimensional implementation of the Rosenbrock function

    Result at (1,1,1) is 0.
    """
    c = parameter.Constant(0, doc='Constant offset, should stay 0')
    l = parameter.List(np.arange(0, 10), repeat=True, doc='list parameter for testing')
    x = parameter.Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='x value of Rosenbrock function')
    y = parameter.Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='y value of Rosenbrock function')
    z = parameter.Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='z value of Rosenbrock function')

    def simulation(self, vector):
        x = np.array(vector)
        simulations = [sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)]
        return simulations



def get_all_samplers():
    def use(cl):
        # Check if class name starts with an exclusion term
        return (inspect.isclass(cl) and
                not any([cl.__name__.startswith(ex) for ex in ('_', 'list')]))

    return inspect.getmembers(spotpy.algorithms, use)


class TestConstantSetups(unittest.TestCase):
    def setUp(self):
        self.setup = RosenbrockWithConstantAndList()

    def sampler_with_constant(self, sampler_class):

        sampler = sampler_class(self.setup, dbformat='ram', save_sim=False)
        sampler.sample(1000)

        self.assertTrue(all(line[1] == 0 for line in sampler.datawriter.ram),
                        msg='Parameter c == 0 not true in all lines with sampler {}'.format(sampler))

    def test_abc_sampler_with_constant(self):
        self.sampler_with_constant(spotpy.algorithms.abc)

    def test_demcz_sampler_with_constant(self):
        self.sampler_with_constant(spotpy.algorithms.demcz)

    def test_dream_sampler_with_constant(self):
        self.sampler_with_constant(spotpy.algorithms.dream)

    def test_fscabc_sampler_with_constant(self):
        self.sampler_with_constant(spotpy.algorithms.fscabc)

    def test_lhs_sampler_with_constant(self):
        self.sampler_with_constant(spotpy.algorithms.lhs)

    def test_mc_sampler_with_constant(self):
        self.sampler_with_constant(spotpy.algorithms.mc)

    def test_mcmc_sampler_with_constant(self):
        self.sampler_with_constant(spotpy.algorithms.mcmc)

    def test_mle_sampler_with_constant(self):
        self.sampler_with_constant(spotpy.algorithms.mle)

    def test_rope_sampler_with_constant(self):
        self.sampler_with_constant(spotpy.algorithms.rope)

    def test_sa_sampler_with_constant(self):
        self.sampler_with_constant(spotpy.algorithms.sa)

    def test_sceua_sampler_with_constant(self):
        self.sampler_with_constant(spotpy.algorithms.sceua)

