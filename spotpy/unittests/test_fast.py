import unittest

try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy


from spotpy.examples.spot_setup_hymod_python import spot_setup



class TestFast(unittest.TestCase):
    def setUp(self):

        self.spot_setup = spot_setup()
        self.rep = 800 # REP must be a multiply of amount of parameters which are in 7 if using hymod
        self.timeout = 10  # Given in Seconds



    def test_fast(self):
        sampler = spotpy.algorithms.fast(self.spot_setup, parallel="seq", dbname='test_FAST', dbformat="ram",
                                          sim_timeout=self.timeout)
        results = []

        sampler.sample(self.rep)
        results = sampler.getdata()
        self.assertEqual(805,len(results))




if __name__ == '__main__':
    unittest.main()
