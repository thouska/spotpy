"""
Tests the various possibilities to create and use parameters

Focus especially the usage of parameters as class attributes
:author: philippkraft
"""
import sys
import unittest
try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy
from spotpy import parameter
import numpy as np
import inspect


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
        return '{}()'.format(type(self).__name__)



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
        return parameter.generate([parameter.Uniform(name, -1, 1) for name in 'abcd'])


class SpotSetupMixedParameterFunction(SpotSetupBase):
    """
    A Test case with two parameters as class parameters (a,b)
    and 2 given from the parameter function
    """
    a = parameter.Uniform(0, 1)
    b = parameter.Uniform(1, 2)

    def parameters(self):
        return parameter.generate([parameter.Uniform(name, -1, 1) for name in 'cd'])


class SpotSetupParameterList(SpotSetupBase):
    """
    A Test case with 4 parameters given from the parameter list
    """
    def __init__(self):
        self.parameters = [parameter.Uniform(name, -1, 1) for name in 'abcd']


class SpotSetupMixedParameterList(SpotSetupBase):
    """
    A Test case with two parameters as class parameters (a,b)
    and 2 given from the parameter function
    """
    a = parameter.Uniform(0, 1)
    b = parameter.Uniform(1, 2)

    def parameters(self):
        return parameter.generate([parameter.Uniform(name, -1, 1) for name in 'cd'])


class TestSetupVariants(unittest.TestCase):
    def setUp(self):
        # Get all Setups from this module
        self.objects = [cls() for cls in SpotSetupBase.get_derived()]

    def test_exists(self):
        self.assertGreater(len(self.objects), 0)

    def parameter_count_test(self, o):
        params = parameter.create_set(o)
        param_names = ','.join(pn for pn in params._fields)
        self.assertEqual(len(params), 4, '{} should have 4 parameters, but found only {} ({})'
                         .format(o, len(params), param_names))
        self.assertEqual(param_names, 'a,b,c,d', '{} Parameter names should be "a,b,c,d" but got "{}"'
                         .format(o, param_names))

    def make_sampler(self, o):
        sampler = spotpy.algorithms.mc(spot_setup=o, dbformat='ram')
        sampler.sample(10)

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


if __name__ == '__main__':
    unittest.main(verbosity=3)

