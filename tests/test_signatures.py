"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska, Philipp Kraft
"""

import unittest

import numpy as np

from spotpy.hydrology.signatures import SignatureMethod


class TestSignatures(unittest.TestCase):
    def setUp(self):

        np.random.seed(0)
        rain = np.random.normal(-1, 5, size=3650)
        rain[rain < 0] = 0.0
        runoff = np.zeros(rain.shape, float)
        stor = 0.0
        for i, prec in enumerate(rain):
            runoff[i] = 0.1 * stor
            stor += prec - runoff[i]

        self.runoff = runoff
        self.rain = rain

    def test_signatures(self):
        sbm_list = SignatureMethod.find_all()

        sig_result = SignatureMethod.run(sbm_list, self.runoff, 1)

        for name, value in sig_result:
            self.assertNotEqual(value, np.nan, "{} returned no value".format(name))


if __name__ == "__main__":

    unittest.main(verbosity=3)
