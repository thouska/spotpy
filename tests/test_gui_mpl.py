# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska, Philipp Kraft
"""

import unittest

import matplotlib

matplotlib.use("Agg")

import inspect
import sys

import numpy as np

from spotpy import parameter
from spotpy.gui.mpl import GUI


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


class SpotSetupMixedParameterFunction(SpotSetupBase):
    """
    A Test case with two parameters as class parameters (a,b)
    and 2 given from the parameter function
    """

    a = parameter.Uniform(0, 1)
    b = parameter.Uniform(1, 2)

    def parameters(self):
        return parameter.generate([parameter.Uniform(name, -1, 1) for name in "cd"])


class TestGuiMpl(unittest.TestCase):
    def test_setup(self):
        setup = SpotSetupMixedParameterFunction()
        with GUI(setup) as gui:
            self.assertTrue(hasattr(gui, "setup"))

    def test_sliders(self):
        setup = SpotSetupMixedParameterFunction()
        with GUI(setup) as gui:
            self.assertEqual(len(gui.sliders), 4)

    def test_clear(self):
        setup = SpotSetupMixedParameterFunction()
        with GUI(setup) as gui:
            gui.clear()
            self.assertEqual(len(gui.lines), 1)

    def test_run(self):
        setup = SpotSetupMixedParameterFunction()
        with GUI(setup) as gui:
            gui.clear()
            gui.run()
            self.assertEqual(len(gui.lines), 2)


if __name__ == "__main__":
    unittest.main()
