import unittest
try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy
from spotpy import parameter
import numpy as np

#https://docs.python.org/3/library/unittest.html

class TestListParameterDistribution(unittest.TestCase):

    def setUp(self):
        self.values = [1, 2, 3, 4, 5]
        self.list_param = parameter.List('test', self.values)
        self.list_param_repeat = parameter.List('test2', self.values, repeat=True)

    def test_list_is_callable(self):
        self.assertTrue(callable(self.list_param), "List instance should be callable")

    def test_list_gives_throwaway_value_on_first_call(self):
        v = self.list_param()
        self.assertNotEqual(self.values[0], v)

    def test_list_gives_1_value_when_size_is_not_specified(self):
        throwaway = self.list_param()
        v = self.list_param()
        self.assertEqual(self.values[0], v)

    def test_list_gives_n_values_when_size_is_n(self):
        throwaway = self.list_param()
        v = self.list_param(len(self.values))
        self.assertEqual(self.values, list(v))


if __name__ == '__main__':
    unittest.main()
