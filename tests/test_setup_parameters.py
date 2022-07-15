"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska, Philipp Kraft

Tests the various possibilities to create and use parameters
Focus especially the usage of parameters as class attributes
"""
import inspect
import sys
import unittest

import numpy as np

import spotpy
from spotpy import parameter


class SpotSetupBase(object):
    """
    The base for a number of test cases.
    Each Test case should have for parameters a,b,c,d and
    the sum of the parameters should be zero
    """

    def simulation(self, par):
        return [par.a + par.b + par.c + par.d]

    def evaluation(self):
        return [0]

    def objectivefunction(self, simulation, evaluation):
        return np.abs(simulation[0] - evaluation[0])

    @classmethod
    def get_derived(cls):
        """
        Returns a list of all derived classes in this module
        """
        module = sys.modules[__name__]

        def predicate(mcls):
            return inspect.isclass(mcls) and issubclass(mcls, cls) and mcls is not cls

        return [mcls for cname, mcls in inspect.getmembers(module, predicate)]

    def __repr__(self):
        return "{}()".format(type(self).__name__)


class SpotSetupClassAttributes(SpotSetupBase):
    """
    A Test case with two parameters as class parameters (a,b)
    and 2 given from the parameter function
    """

    a = parameter.Uniform(-1, 1)
    b = parameter.Uniform(-1, 1)
    c = parameter.Uniform(-1, 1)
    d = parameter.Uniform(-1, 1)


class SpotSetupParameterFunction(SpotSetupBase):
    """
    A Test case with 4 parameters given from the parameter function
    """

    def parameters(self):
        return parameter.generate([parameter.Uniform(name, -1, 1) for name in "abcd"])


class SpotSetupMixedParameterFunction(SpotSetupBase):
    """
    A Test case with two parameters as class parameters (a,b)
    and 2 given from the parameter function
    """

    a = parameter.Uniform(0, 1)
    b = parameter.Uniform(1, 2)

    def parameters(self):
        return parameter.generate([parameter.Uniform(name, -1, 1) for name in "cd"])


class SpotSetupParameterList(SpotSetupBase):
    """
    A Test case with 4 parameters given from the parameter list
    """

    def __init__(self):
        self.parameters = [parameter.Uniform(name, -1, 1) for name in "abcd"]


class SpotSetupMixedParameterList(SpotSetupBase):
    """
    A Test case with two parameters as class parameters (a,b)
    and 2 given from the parameter function
    """

    a = parameter.Uniform(0, 1)
    b = parameter.Uniform(1, 2)

    def parameters(self):
        return parameter.generate([parameter.Uniform(name, -1, 1) for name in "cd"])


class TestParameterSet(unittest.TestCase):
    def setUp(self):
        model = SpotSetupParameterFunction()
        param_info = model.parameters()
        self.ps = parameter.ParameterSet(param_info)

    def test_create(self):
        self.assertEqual(type(self.ps), parameter.ParameterSet)

    def test_assign(self):
        values = [1] * len(self.ps)
        self.ps(*values)
        self.assertEqual(list(self.ps), values)
        # Test if wrong number of parameters raises
        with self.assertRaises(ValueError):
            self.ps(*values[:-1])

    def test_iter(self):
        values = [1] * len(self.ps)
        self.ps(*values)
        ps_values = list(self.ps)
        self.assertEqual(values, ps_values)

    def test_getitem(self):
        values = [1] * len(self.ps)
        self.ps(*values)
        self.assertEqual(self.ps["a"], 1.0)
        self.assertEqual(self.ps[0], 1.0)

    def test_getattr(self):
        values = [1] * len(self.ps)
        self.ps(*values)

        with self.assertRaises(AttributeError):
            _ = self.ps.__x

        self.assertEqual(self.ps.a, 1.0)
        self.assertEqual(
            list(self.ps.random),
            list(self.ps),
            "Access to random variable does not equal list of names",
        )

        with self.assertRaises(AttributeError):
            _ = self.ps.x

    def test_setattr(self):
        self.ps.a = 2
        self.assertEqual(self.ps[0], 2)

    def test_dir(self):
        values = [1] * len(self.ps)
        self.ps(*values)

        attrs = dir(self.ps)
        for param in self.ps.name:
            self.assertIn(
                param, attrs, "Attribute {} not found in {}".format(param, self.ps)
            )
        for prop in ["maxbound", "minbound", "name", "optguess", "random", "step"]:
            self.assertIn(
                prop, attrs, "Property {} not found in {}".format(prop, self.ps)
            )

    def test_str(self):
        values = [1] * len(self.ps)
        self.ps(*values)
        self.assertEqual(str(self.ps), "parameters(a=1, b=1, c=1, d=1)")

    def test_repr(self):
        values = [1] * len(self.ps)
        self.ps(*values)
        self.assertEqual(repr(self.ps), "spotpy.parameter.ParameterSet()")


class TestSetupVariants(unittest.TestCase):
    def setUp(self):
        # Get all Setups from this module
        self.objects = [cls() for cls in SpotSetupBase.get_derived()]

    def test_exists(self):
        self.assertGreater(len(self.objects), 0)

    def parameter_count_test(self, o):
        params = parameter.create_set(o, valuetype="optguess")
        param_names = ",".join(pn for pn in params.name)
        self.assertEqual(
            len(params),
            4,
            "{} should have 4 parameters, but found only {} ({})".format(
                o, len(params), param_names
            ),
        )
        self.assertEqual(
            param_names,
            "a,b,c,d",
            '{} Parameter names should be "a,b,c,d" but got "{}"'.format(
                o, param_names
            ),
        )

    def make_sampler(self, o, algo=spotpy.algorithms.mc):
        sampler = algo(spot_setup=o, dbformat="ram")
        sampler.sample(100)

    def test_parameter_class(self):
        self.parameter_count_test(SpotSetupClassAttributes())

    def test_parameter_function(self):
        self.parameter_count_test(SpotSetupParameterFunction())

    def test_parameter_list(self):
        self.parameter_count_test(SpotSetupParameterList())

    def test_parameter_mixed_list(self):
        self.parameter_count_test(SpotSetupMixedParameterList())

    def test_parameter_mixed_function(self):
        self.parameter_count_test(SpotSetupMixedParameterFunction())

    def test_sampler(self):
        for o in self.objects:
            self.make_sampler(o)

    def test_abc_sampler(self):
        for o in self.objects:
            self.make_sampler(o, spotpy.algorithms.abc)

    def test_demcz_sampler(self):
        for o in self.objects:
            self.make_sampler(o, spotpy.algorithms.demcz)

    def test_dream_sampler(self):
        for o in self.objects:
            self.make_sampler(o, spotpy.algorithms.dream)

    def test_fscabc_sampler(self):
        for o in self.objects:
            self.make_sampler(o, spotpy.algorithms.fscabc)

    def test_lhs_sampler(self):
        for o in self.objects:
            self.make_sampler(o, spotpy.algorithms.lhs)

    def test_mc_sampler(self):
        for o in self.objects:
            self.make_sampler(o, spotpy.algorithms.mc)

    def test_mcmc_sampler(self):
        for o in self.objects:
            self.make_sampler(o, spotpy.algorithms.mcmc)

    def test_mle_sampler(self):
        for o in self.objects:
            self.make_sampler(o, spotpy.algorithms.mle)

    def test_rope_sampler(self):
        for o in self.objects:
            self.make_sampler(o, spotpy.algorithms.rope)

    def test_sa_sampler(self):
        for o in self.objects:
            self.make_sampler(o, spotpy.algorithms.sa)

    def test_sceua_sampler(self):
        for o in self.objects:
            self.make_sampler(o, spotpy.algorithms.sceua)


if __name__ == "_main__":
    unittest.main(verbosity=3)
