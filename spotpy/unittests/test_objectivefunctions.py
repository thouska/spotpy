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

        print(self.simulation)
        print(self.evaluation)

    def test_bias(self):
        res = of.bias(self.evaluation, self.simulation)
        self.assertAlmostEqual(res, 1.2387193462811703, self.tolerance)

    def test_length_mismatch_return_nan(self):
        all_funcs = of._all_functions

        for func in all_funcs:
            res = func([0], [0, 1])
            self.assertIs(res, np.nan, "Expected np.nan in length mismatch, Got {}".format(res))


if __name__ == '__main__':
    unittest.main()
