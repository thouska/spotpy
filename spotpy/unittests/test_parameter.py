import unittest
try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy
from spotpy import parameter
import numpy as np

from testutils import repeat

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
        _ = self.list_param()
        v = self.list_param()
        self.assertEqual(self.values[0], v)

    def test_list_gives_n_values_when_size_is_n(self):
        _ = self.list_param()
        v = self.list_param(len(self.values))
        self.assertEqual(self.values, list(v))

    def test_list_gives_cycled_values_with_repeat(self):
        _ = self.list_param_repeat()
        v1 = self.list_param_repeat()
        for k in range(len(self.values) - 1):
            self.list_param_repeat()

        v2 = self.list_param_repeat()

        self.assertEqual(v1, v2)

    def test_list_gives_cycled_values_with_repeat_and_size(self):
        _ = self.list_param_repeat()
        v1 = self.list_param_repeat(len(self.values))
        v2 = self.list_param_repeat(len(self.values))

        self.assertEqual(list(v1), list(v2))

    def test_list_without_repeat_raises_index_error_on_invalid_extra_call(self):
        _ = self.list_param()
        for _ in range(len(self.values)):
            self.list_param()

        self.assertRaises(IndexError, self.list_param)

    def test_astuple(self):
        _ = self.list_param()
        v, name, step, optguess, minbound, maxbound = self.list_param.astuple()
        self.assertEqual(self.values[0], v)
        self.assertEqual("test", name)

        # the values of step, optguess, minbound and maxbound don't matter

class TestUniformParameterDistribution(unittest.TestCase):

    # Relatively low tolerance because it's a probabilistic distribution
    tolerance = 1

    def setUp(self):
        pass


    def test_uniform_is_callable(self):
        unif = parameter.Uniform("test", low=0, high=1)
        self.assertTrue(callable(unif), "Uniform param instance should be callable")

    def test_uniform_has_correct_bounds(self):
        unif = parameter.Uniform("test", low=0, high=1)
        self.assertGreaterEqual(unif.minbound, 0)
        self.assertLessEqual(unif.maxbound, 1)

    def test_uniform_processes_non_keyword_args(self):
        unif = parameter.Uniform("test", 0, 1)

    @repeat(10)
    def test_uniform_has_correct_statistics(self):
        unif = parameter.Uniform("test", 0, 1)
        # Generate 10k random numbers
        nums = [unif() for _ in range(10000)]
        self.assertAlmostEqual(np.mean(nums), (1 - 0)/2.0, self.tolerance, "Mean of Unif(0, 1) should be 1/2")
        self.assertAlmostEqual(np.var(nums), 1.0/12, self.tolerance, "Variance of Unif(0, 1) should be 1/12")


class TestConstantParameterDistribution(unittest.TestCase):

    def setUp(self):
        self.const = parameter.Constant("test", 10)

    def test_constant_is_callable(self):
        self.assertTrue(callable(self.const), "Constant param instance should be callable")

    def test_constant_gives_only_constants(self):
        nums = set([self.const() for _ in range(1000)])
        self.assertEqual(len(nums), 1)
        self.assertEqual(nums.pop(), 10)

class TestNormalParameterDistribution(unittest.TestCase):

    # Relatively low tolerance because it's a probabilistic distribution
    tolerance = 0

    def setUp(self):
        self.norm = parameter.Normal("test", mean=5, stddev=10)

    def test_normal_is_callable(self):
        self.assertTrue(callable(self.norm), "Normal param instance should be callable")

    def test_normal_processes_non_keyword_args(self):
        _ = parameter.Normal("test", 0, 1)

    @repeat(5)
    def test_normal_has_correct_statistics(self):
        nums = [self.norm() for _ in range(10000)]
        self.assertAlmostEqual(np.mean(nums), 5, self.tolerance, "Mean of Norm(5, 10) should be 5")
        self.assertAlmostEqual(np.std(nums), 10, self.tolerance, "SD of Norm(5, 10) should be 10")

class TestLogNormalParameterDistribution(unittest.TestCase):

    # Relatively low tolerance because it's a probabilistic distribution
    tolerance = 0

    def setUp(self):
        self.log_norm = parameter.logNormal("test", mean=5, sigma=10)

    def test_normal_is_callable(self):
        self.assertTrue(callable(self.log_norm), "Log Normal param instance should be callable")

    def test_normal_processes_non_keyword_args(self):
        _ = parameter.logNormal("test", 0, 1)

    @repeat(5)
    def test_normal_has_correct_statistics(self):
        nums = [self.log_norm() for _ in range(10000)]
        log_nums = np.log(nums)
        self.assertAlmostEqual(np.mean(log_nums), 5, self.tolerance, "Mean of Log(LogNorm(5, 10)) should be 5")
        self.assertAlmostEqual(np.std(log_nums), 10, self.tolerance, "SD of Log(LogNorm(5, 10)) should be 10")


class TestChiSquareParameterDistribution(unittest.TestCase):
    # Relatively low tolerance because it's a probabilistic distribution
    tolerance = 0

    def setUp(self):
        self.chisq = parameter.Chisquare("test", dt=5)

    def test_chisq_is_callable(self):
        self.assertTrue(callable(self.chisq), "Chisquare param instance should be callable")

    def test_chisq_processes_non_keyword_args(self):
        _ = parameter.Chisquare("test", 5)

    @repeat(5)
    def test_chisq_has_correct_statistics(self):
        nums = [self.chisq() for _ in range(10000)]
        self.assertAlmostEqual(np.mean(nums), 5, self.tolerance, "Mean of Chisquare(5) should be 5")
        self.assertAlmostEqual(np.std(nums), np.sqrt(2*5), self.tolerance, "SD of Chisquare(5) should be sqrt(2*5)")

if __name__ == '__main__':
    unittest.main()
