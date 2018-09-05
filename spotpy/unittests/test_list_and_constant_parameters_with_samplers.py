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


class RosenbrockWithConstant(Rosenbrock):
    """
    A 3 dimensional implementation of the Rosenbrock function

    Result at (1,1,1) is 0.
    """
    x = parameter.Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='x value of Rosenbrock function')
    y = parameter.Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='y value of Rosenbrock function')
    z = parameter.Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='z value of Rosenbrock function')
    c = parameter.Constant(0, doc='Constant offset, should stay 0')

    def simulation(self, vector):

        simulations = [v + vector[-1] for v in super().simulation(vector[:-1])]
        return simulations


class RosenbrockWithList(Rosenbrock):
    """
    A 3 dimensional implementation of the Rosenbrock function

    Result at (1,1,1) is 0.
    """
    x = parameter.Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='x value of Rosenbrock function')
    y = parameter.Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='y value of Rosenbrock function')
    z = parameter.Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='z value of Rosenbrock function')
    l = parameter.List(range(10), repeat=True)


def get_all_samplers():
    def use(cl):
        # Check if class name starts with an exclusion term
        return (inspect.isclass(cl) and
                not any([cl.__name__.startswith(ex) for ex in ('_', 'list')]))

    return inspect.getmembers(spotpy.algorithms, use)


class TestConstantSetups(unittest.TestCase):
    def setUp(self):
        self.setup = RosenbrockWithConstant()

    def sampler_with_constant(self, sampler_class):

        sampler_name = sampler_class.__name__
        print(sampler_name)
        sampler = sampler_class(self.setup, dbformat='ram', save_sim=False)
        sampler.sample(100)
        print(sampler.datawriter.ram[-1])
        print('-' * 50)
        self.assertTrue(all(line[-2] == 0 for line in sampler.datawriter.ram),
                        msg='Parameter c == 0 not true in all lines')

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


class TestListSetups(unittest.TestCase):

    def setUp(self):
        self.setup = RosenbrockWithList()
        print(spotpy.describe.setup(self.setup))


    def test_list_parameter(self):

        for sampler_name, sampler_class in get_all_samplers():
            if parameter.List in sampler_class._excluded_parameter_classes:
                print(sampler_name, ' should raise')
                with self.assertRaises(TypeError, msg="Sampler {} did not raise TypeError for List parameter"):
                    _ = sampler_class(self.setup, dbformat='ram')
            else:
                print(sampler_name, ' should not raise')
                sampler = sampler_class(self.setup, dbformat='ram')
                sampler.sample(10)
