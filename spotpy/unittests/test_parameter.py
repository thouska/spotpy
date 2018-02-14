import unittest
try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy
from spotpy import parameter
import numpy as np

# Import inspect to scan spotpy.parameter for all Parameter classes
import inspect
from testutils import repeat

# https://docs.python.org/3/library/unittest.html


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
        _ = parameter.Uniform("test", 0, 1)

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
        self.df = 5
        self.chisq = parameter.Chisquare("test", dt=self.df)

    def test_chisq_is_callable(self):
        self.assertTrue(callable(self.chisq), "Chisquare param instance should be callable")

    def test_chisq_processes_non_keyword_args(self):
        _ = parameter.Chisquare("test", 5)

    @repeat(5)
    def test_chisq_has_correct_statistics(self):
        nums = [self.chisq() for _ in range(10000)]
        self.assertAlmostEqual(np.mean(nums), self.df, self.tolerance, 
                               "Mean of Chisquare({df}) should be {df}".format(df=self.df))
        self.assertAlmostEqual(np.std(nums), np.sqrt(2*self.df), self.tolerance, 
                               "SD of Chisquare({df}) should be sqrt(2*{df})".format(df=self.df))


class TestExponentialParameterDistribution(unittest.TestCase):
    # Relatively low tolerance because it's a probabilistic distribution
    tolerance = 0

    def setUp(self):
        self.beta = 5
        self.exp = parameter.Exponential("test", scale=self.beta)

    def test_exp_is_callable(self):
        self.assertTrue(callable(self.exp), "Exponential param instance should be callable")

    def test_exp_processes_non_keyword_args(self):
        _ = parameter.Exponential("test", self.beta)

    @repeat(5)
    def test_exp_has_correct_statistics(self):
        nums = [self.exp() for _ in range(10000)]
        self.assertAlmostEqual(np.mean(nums), self.beta, self.tolerance, 
                               "Mean of Exponential({beta}) should be {beta}".format(beta=self.beta))
        self.assertAlmostEqual(np.std(nums), self.beta, self.tolerance, 
                               "SD of Exponential({beta}) should be {beta}".format(beta=self.beta))


class TestGammaParameterDistribution(unittest.TestCase):
    # Relatively low tolerance because it's a probabilistic distribution
    tolerance = 0

    def setUp(self):
        self.shape = 5
        self.scale = 1.2
        self.gamma = parameter.Gamma("test", shape=self.shape, scale=self.scale)

    def test_gamma_is_callable(self):
        self.assertTrue(callable(self.gamma), "Gamma param instance should be callable")

    def test_gamma_processes_non_keyword_args(self):
        _ = parameter.Gamma("test", self.shape, self.scale)

    @repeat(5)
    def test_gamma_has_correct_statistics(self):
        nums = [self.gamma() for _ in range(10000)]
        expected_mean = self.shape*self.scale
        expected_sd = np.sqrt(self.shape*self.scale*self.scale)
        self.assertAlmostEqual(np.mean(nums), expected_mean, self.tolerance, 
                               "Mean of Gamma({}, {}) should be {}".format(self.shape, self.scale, expected_mean))
        self.assertAlmostEqual(np.std(nums), expected_sd, self.tolerance, 
                               "SD of Gamma({}, {}) should be {}".format(self.shape, self.scale, expected_sd))


class TestWaldParameterDistribution(unittest.TestCase):
    # Relatively low tolerance because it's a probabilistic distribution
    tolerance = 0

    def setUp(self):
        self.mean = 5
        self.scale = 1.2
        self.wald = parameter.Wald("test", mean=self.mean, scale=self.scale)

    def test_wald_is_callable(self):
        self.assertTrue(callable(self.wald), "Wald param instance should be callable")

    def test_wald_processes_non_keyword_args(self):
        _ = parameter.Wald("test", self.mean, self.scale)

    def test_wald_has_correct_statistics(self):
        nums = [self.wald() for _ in range(40000)]
        expected_sd = np.sqrt(self.mean**3 / self.scale)
        self.assertAlmostEqual(np.mean(nums), self.mean, self.tolerance, 
                               "Mean of Wald({}, {}) should be {}".format(self.mean, self.scale, self.mean))
        self.assertAlmostEqual(np.std(nums), expected_sd, self.tolerance, 
                               "SD of Wald({}, {}) should be {}".format(self.mean, self.scale, expected_sd))


class TestWeibullParameterDistribution(unittest.TestCase):
    # Relatively low tolerance because it's a probabilistic distribution
    tolerance = 0

    def setUp(self):
        self.a = 5
        self.weibull = parameter.Weibull("test", a=self.a)

    def test_weibull_is_callable(self):
        self.assertTrue(callable(self.weibull), "Weibull param instance should be callable")

    def test_weibull_processes_non_keyword_args(self):
        _ = parameter.Weibull("test", self.a)

    def test_weibull_has_correct_statistics(self):
        nums = [self.weibull() for _ in range(10000)]
        self.assertAlmostEqual(np.mean(nums), 0.918169, self.tolerance, 
                               "Mean of Weibull({}) should be {}".format(self.a, 0.918169))
        self.assertAlmostEqual(np.std(nums), 0.0442300, self.tolerance, 
                               "SD of Weibull({}) should be {}".format(self.a, 0.0442300))


class TestTriangularParameterDistribution(unittest.TestCase):
    # Relatively low tolerance because it's a probabilistic distribution
    tolerance = 0

    def setUp(self):
        self.a, self.c, self.b = 0, 2, 5
        self.triangular = parameter.Triangular("test", left=self.a, mode=self.c, right=self.b)

    def test_triangular_is_callable(self):
        self.assertTrue(callable(self.triangular), "Triangular param instance should be callable")

    def test_triangular_has_correct_statistics(self):
        nums = [self.triangular() for _ in range(10000)]
        expected_mean = (self.a + self.b + self.c) / 3
        expected_sd = np.sqrt((self.a**2 + self.b**2 + self.c**2 - self.a*self.c - self.a*self.b - self.b*self.c)/18)
        self.assertAlmostEqual(np.mean(nums), expected_mean, self.tolerance, 
                               "Mean of Triangular({}, {}, {}) should be {}"
                               .format(self.a, self.c, self.b, expected_mean))
        self.assertAlmostEqual(np.std(nums), expected_sd, self.tolerance, 
                               "SD of Triangular({}, {}, {}) should be {}"
                               .format(self.a, self.c, self.b, expected_sd))


class TestParameterArguments(unittest.TestCase):
    """
    Test by philippkraft
    checks behaviour of Uniform and Triangular for a number of
    different arguments.
    """

    def setUp(self):
        """
        Setup for 2 simple parameter cases
        :return:
        """
        self.classes = [parameter.Uniform, parameter.Triangular]
        self.rndargs = [(10, 20), (10, 15, 20)]

    def test_correct_with_default_step(self):
        for cl, args in zip(self.classes, self.rndargs):
            p_no_name = cl(*args)
            p_with_name = cl(cl.__name__, *args)
            self.assertTrue(10 <= p_no_name.optguess < 20, 'Optguess out of boundaries')
            self.assertTrue(10 <= p_with_name.optguess < 20, 'Optguess out of boundaries')
            self.assertTrue(p_no_name.step < 10, 'Step to large')
            self.assertTrue(p_with_name.step < 10, 'Step to large')

    def test_correct_with_extra_args(self):
        for cl, args in zip(self.classes, self.rndargs):
            p_no_name = cl(*args, step=1, default=12)
            p_with_name = cl(cl.__name__, *args, step=1, default=12)
            self.assertTrue(p_no_name.optguess == 12,
                            'Optguess not found from default (name={})'.format(repr(p_no_name.name)))
            self.assertTrue(p_with_name.optguess == 12,
                            'Optguess not found from default (name={})'.format(repr(p_with_name.name)))
            self.assertTrue(p_no_name.step == 1,
                            'Step overridden by class (name={})'.format(repr(p_no_name.name)))
            self.assertTrue(p_with_name.step == 1,
                            'Step overridden by class (name={})'.format(repr(p_with_name.name)))


def make_args(pcls):
    """
    Returns an args tuple for the parameter class pcls.
    """
    return tuple(range(1, len(pcls.__rndargs__) + 1))


def get_classes():
    """
    Get all classes from spotpy.parameter module, except special cases
    """    
    def predicate(cls):
        return (inspect.isclass(cls)
                and cls not in [parameter.Base, parameter.List]
                and issubclass(cls, parameter.Base))
    
    return [[cname, cls]
            for cname, cls in
            inspect.getmembers(parameter, predicate)
            ]


class TestParameterClasses(unittest.TestCase):
    """
    Test by philippkraft to test all available Parameter classes, except List
    """

    def test_classes_available(self):
        classes = get_classes()
        self.assertGreaterEqual(len(classes), 1, 'No parameter classes found in spotpy.parameter')
        self.assertIn('Uniform', [n for n, c in classes], 'Parameter class spotpy.parameter.Uniform not found')

    def test_create_posargs(self):
        """
        Checks if the right number of arguments is present
        :return:
        """
        for cname, cls in get_classes():
            # Add step parameter to args
            args = make_args(cls) + (0.01,)
            # Test with name
            p_name = cls(cname, *args)
            # Test without name
            p_no_name = cls(*args)
            # Check the names
            self.assertEqual(p_name.name, cname)
            # Check name is empty, when no name is given
            self.assertEqual(p_no_name.name, '')
            # Test Parameter character for both
            for p in [p_name, p_no_name]:
                self.assertTrue(callable(p))
                self.assertGreater(p(), -np.inf)
                self.assertTrue(p.step == 0.01,
                                '{} did not receive step as the correct value, check number of arguments'
                                .format(cls.__name__))

    def test_create_kwargs(self):
        """
        Check keyword arguments for distribution function
        :return:
        """
        for cname, cls in get_classes():
            kwargs = dict(zip(cls.__rndargs__, make_args(cls)))
            # Test with name
            p_name = cls(cname, step=0.01, **kwargs)
            # Test without name
            p_no_name = cls(step=0.01, **kwargs)
            # Check the names
            self.assertEqual(p_name.name, cname)
            self.assertEqual(p_no_name.name, '')
            # Test Parameter character for both
            for p in [p_name, p_no_name]:
                self.assertTrue(callable(p))
                self.assertGreater(p(), -np.inf)
                self.assertEqual(p.step, 0.01,
                                 '{} did not receive step as the correct value, check number of arguments'
                                 .format(cls.__name__))

    def test_too_many_args(self):
        """
        Check if class raises when too many arguments are given
        """
        for cname, cls in get_classes():
            # Implicit definition of step in args
            step_args = make_args(cls) + (1, )
            with self.assertRaises(TypeError):
                _ = cls(*step_args, step=1)
            with self.assertRaises(TypeError):
                _ = cls(cls.__name__, *step_args, step=1)

    def test_too_few_args(self):
        """
        Check if class raises when too few arguments are given
        """
        for cname, cls in get_classes():
            args = make_args(cls)
            # Double definition of step
            with self.assertRaises(TypeError):
                _ = cls(*args[:-1], step=1)
            with self.assertRaises(TypeError):
                _ = cls(cls.__name__, *args[:-1], step=1)


if __name__ == '__main__':
    unittest.main()
