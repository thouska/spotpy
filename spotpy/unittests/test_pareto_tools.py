import numpy as np
import os
import unittest

try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy

from spotpy.pareto_tools import crowd_dist
from pprint import pprint
import pandas as pd

def generate_multi_array():
    len_x, len_y = np.random.randint(1,100, 2)
    return np.random.normal(21.105,42,len_x*len_y).reshape((len_x,len_y))

def generate_array_testfiles():
    for j in range(1000):
        m_arr = generate_multi_array()
        path = os.path.abspath(os.path.dirname(__file__))
        folder = os.path.join(path,"pareto_tools/crowd_dist_test")
        filename = f"test_file_{j:05d}"
        testfile = os.path.join(folder,filename)
        np.savetxt(testfile,m_arr)



class TestParetoTools(unittest.TestCase):
    def setUp(self):
        self.pareto_folder = "pareto_tools/crowd_dist_test"

    def test_crowd_dist_matlab(self):
        for j in range(1,1000): # TODO loop from 1 to 999 (1000)
            print("### TEST No "+str(j)+" of " + str(1000)+" ###", end="\r")
            path = os.path.abspath(os.path.dirname(__file__))
            folder = os.path.join(path, self.pareto_folder)

            filename = f"test_file_{j:05d}"
            testfile = os.path.join(folder, filename)

            resultfile = os.path.join(folder, f"test_file_{j:05d}_result")

            m_arr = pd.read_table(testfile,header=None,sep='\s+').values
            matlab_result = np.loadtxt(resultfile)

            python_result = crowd_dist(m_arr)
            diff = np.abs(python_result - matlab_result)
            if not (diff < 5e-15).all():
                self.fail("Difference in results detected: "+ str(np.max(diff)))



unittest.main()