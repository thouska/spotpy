"""Utility functions for unit test execution"""

def repeat(times):
    """Repeats a single test the given number of times

    Usage:
    @repeat(5)
    def test_foo(self):
        self.assertTrue(self.bar == self.baz)

    The above test will execute 5 times

    Reference: https://stackoverflow.com/a/13606054/4014685
    """
    def repeatHelper(f):
        def func_repeat_executor(*args, **kwargs):
            for i in range(0, times):
                f(*args, **kwargs)
                args[0].setUp()

        return func_repeat_executor

    return repeatHelper
