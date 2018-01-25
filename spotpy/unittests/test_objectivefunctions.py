import unittest
from spotpy import objectivefunctions as of
import numpy as np

#https://docs.python.org/3/library/unittest.html

class TestObjectiveFunctions(unittest.TestCase):

    # How many digits to match in case of floating point answers
    tolerance = 10

    def setUp(self):
        np.random.seed(42)
        self.simulation = np.random.randn(10)
        self.evaluation = np.random.randn(10)

    def test_bias(self):
        res = of.bias(self.evaluation, self.simulation)
        self.assertAlmostEqual(res, 1.2387193462811703, self.tolerance)

    def test_pbias(self):
        res = of.pbias(self.evaluation, self.simulation)
        self.assertAlmostEqual(res, -156.66937901878677, self.tolerance)

    def test_nashsutcliffe(self):
        res = of.nashsutcliffe(self.evaluation, self.simulation)
        self.assertAlmostEqual(res, -4.1162070769985508, self.tolerance)

    def test_lognashsutcliffe(self):
        # Since log is invalid for negative numbers:
        res = of.lognashsutcliffe(self.evaluation + 3, self.simulation + 3)
        self.assertAlmostEqual(res, -2.3300973555530344, self.tolerance)

    def test_lognashsutcliffe_invalid_obs(self):
        res = of.lognashsutcliffe(self.evaluation, self.simulation)
        self.assertTrue(np.isnan(res))

    def test_length_mismatch_return_nan(self):
        all_funcs = of._all_functions

        for func in all_funcs:
            res = func([0], [0, 1])
            self.assertTrue(np.isnan(res), "Expected np.nan in length mismatch, Got {}".format(res))


if __name__ == '__main__':
    unittest.main()
