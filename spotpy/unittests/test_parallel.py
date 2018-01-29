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

    def test_sequential(self):
        results = []
        sampler = spotpy.algorithms.mc(self.spot_setup, parallel="seq",
                                       dbname='test_parallel_test_sequential', dbformat="csv",
                                       sim_timeout=self.timeout)
        sampler.sample(self.rep)
        results.append(sampler.getdata())
        self.assertEqual(self.rep,len(results[0]))

    def test_mproc(self):
        results = []
        sampler = spotpy.algorithms.rope(self.spot_setup, parallel="mpc",
                                       dbname='test_parallel_test_mproc', dbformat="ram",
                                       sim_timeout=self.timeout)
        sampler.sample(200)
        results.append(sampler.getdata())
        self.assertEqual(len(results[0]), 204)

    def test_umproc(self):
        results = []
        sampler = spotpy.algorithms.rope(self.spot_setup, parallel="umpc",
                                       dbname='test_parallel_test_umproc', dbformat="ram",
                                       sim_timeout=self.timeout)
        sampler.sample(200)
        results.append(sampler.getdata())
        self.assertEqual(len(results[0]),204)


if __name__ == '__main__':
    unittest.main()
