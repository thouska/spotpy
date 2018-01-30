import unittest

try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy


from spotpy.examples.spot_setup_hymod_python import spot_setup
import numpy as np


class TestParallel(unittest.TestCase):
    def setUp(self):

        self.spot_setup = spot_setup()
        self.rep = 200
        self.timeout = 10  # Given in Seconds
        import sys
        self.use_py_3 = sys.version_info[0] >= 3


    def test_sequential(self):
        results = []
        sampler = spotpy.algorithms.mc(self.spot_setup, parallel="seq",
                                       dbname='test_parallel_test_sequential', dbformat="csv",
                                       sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results.append(sampler.getdata())
        if self.use_py_3:
            self.assertEqual(self.rep, len(results[0]))
        else:
            self.assertGreaterEqual(len(results[0]),100)

    def test_mproc(self):
        results = []
        sampler = spotpy.algorithms.rope(self.spot_setup, parallel="mpc",
                                       dbname='test_parallel_test_mproc', dbformat="ram",
                                       sim_timeout=self.timeout)
        sampler.sample(100)
        results.append(sampler.getdata())
        if self.use_py_3:
            self.assertEqual(len(results[0]), 98)
        else:
            self.assertGreaterEqual(len(results[0]), 90)


    def test_umproc(self):
        results = []
        sampler = spotpy.algorithms.rope(self.spot_setup, parallel="umpc",
                                       dbname='test_parallel_test_umproc', dbformat="ram",
                                       sim_timeout=self.timeout)
        sampler.sample(100)
        results.append(sampler.getdata())
        if self.use_py_3:
            self.assertEqual(len(results[0]), 98)
        else:
            self.assertGreaterEqual(len(results[0]), 90)




if __name__ == '__main__':
    unittest.main()
