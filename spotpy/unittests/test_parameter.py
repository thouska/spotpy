import unittest
try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy
from spotpy import parameter

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

class TestParameterArguments(unittest.TestCase):

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
            self.assertTrue(p_with_name.step < 10 , 'Step to large')

    def test_correct_with_extra_args(self):
        for cl, args in zip(self.classes, self.rndargs):
            p_no_name = cl(*args, step=1, default=12)
            p_with_name = cl(cl.__name__, *args, step=1, default=12)
            self.assertTrue(p_no_name.optguess == 12, 'Optguess not found from default (name={})'.format(repr(p_no_name.name)))
            self.assertTrue(p_with_name.optguess == 12, 'Optguess not found from default (name={})'.format(repr(p_with_name.name)))
            self.assertTrue(p_no_name.step == 1, 'Step overridden by class (name={})'.format(repr(p_no_name.name)))
            self.assertTrue(p_with_name.step == 1, 'Step overridden by class (name={})'.format(repr(p_with_name.name)))

    def test_too_many_args(self):
        for cl, args in zip(self.classes, self.rndargs):
            # Double definition of step
            step_args = args + (1, )
            with self.assertRaises(TypeError):
                p_no_name = cl(*step_args, step=1)
            with self.assertRaises(TypeError):
                p_with_name = cl(cl.__name__, *step_args, step=1)

    def test_too_few_args(self):
        for cl, args in zip(self.classes, self.rndargs):
            # Double definition of step
            with self.assertRaises(TypeError):
                p_no_name = cl(*args[:-1], step=1)

            with self.assertRaises(TypeError):
                p_with_name = cl(cl.__name__, *args[:-1], step=1)

class TestParameterClasses(unittest.TestCase):

    def setUp(self):
        """
        Get all classes from spotpy.parameter module, except special cases
        """
        self.classes = []
        for cname, cls in vars(parameter).items():
            if (cls is type
                    and cls not in [parameter.Base, parameter.List]
                    and issubclass(cls, parameter.Base)):
                self.classes.append(cls)


    def test_create_posargs(self):
        """
        Checks if the right number of arguments is present
        :return:
        """
        for cname, cls in self.classes:
            args = tuple(range(1, len(cls.__rndargs__) + 1)) + (0.01,)
            p = cls(cname, *args)
            self.assertFalse(p.name)
            self.assertTrue(callable(p))
            a = p()
            self.assertTrue(p.step == 0.01,
                            '{} did not receive step as the correct value, check number of arguments'
                            .format(cls.__name__))


    def test_create_kwargs(self):
        """
        Check keyword arguments for distribution function
        :return:
        """
        for cname, cls in self.classes:
            kwargs = dict((kw, i+1) for i, kw in enumerate(cls.__rndargs__))
            p = cls(cname, **kwargs)
            self.assertFalse(p.name)
            self.assertTrue(callable(p))

if __name__ == '__main__':
    unittest.main()
