import unittest
import spotpy.signatures as sig
import numpy as np

#https://docs.python.org/3/library/unittest.html

class TestSignatures(unittest.TestCase):

    def setUp(self):
        self.data = np.random.gamma(0.7,2,500)


    def test_getBaseflowIndex(self):
        b = sig.getMeanFlow(self.data)
        self.assertTrue(np.abs(b-1.5) < 0.5)

    def test_getCoeffVariation(self):
        b = sig.getCoeffVariation(self.data)
        self.assertTrue(np.abs(b-1.16) < 0.1)


if __name__ == '__main__':
    unittest.main()