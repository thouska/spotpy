import unittest
import os
import numpy as np

try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy

from spotpy.pareto_tools import crowd_dist
from spotpy.pareto_tools import nd_check
import pandas as pd


def generate_multi_array(y_len=None):
    len_x, len_y = np.random.randint(1,100, 2)
    if y_len is not None:
        len_y = y_len
    return np.random.normal(21.105,42,len_x*len_y).reshape((len_x,len_y))

def generate_array_testfiles():
    for j in range(1000):
        m_arr = generate_multi_array()
        path = os.path.abspath(os.path.dirname(__file__))
        folder = os.path.join(path,"pareto_tools/crowd_dist_test")
        filename = "test_file_%05d" % j
        testfile = os.path.join(folder,filename)
        np.savetxt(testfile,m_arr)

def generate_nd_check_array_testfiles():
    for j in range(1000):
        num_objs, num_dec, height = np.random.randint(1, 100, 3)

        Jtest = np.random.normal(0, 1, num_objs)  # reshape((len_x, len_y)) randn(1, num_objs);
        stest = np.random.normal(0, 1, num_dec)  # randn(1, num_dec);

        PF_set = np.random.normal(21, 42, (num_objs + num_dec) * height).reshape(height, (num_objs + num_dec))

        path = os.path.abspath(os.path.dirname(__file__))
        folder = os.path.join(path,"pareto_tools/nd_check_test")

        np.savetxt(os.path.join(folder, "test_%05d_PF_set" % j), PF_set)
        np.savetxt(os.path.join(folder, "test_%05d_stest" % j), stest)
        np.savetxt(os.path.join(folder, "test_%05d_Jtest" % j), Jtest)

indicies = range(1, 1000)

class ParetoTestsContainer(unittest.TestCase):
    def setUp(self):
        self.pareto_folder = "pareto_tools/crowd_dist_test"

    def test_nd_check_pytest_wrapper(self):
        for j in indicies:
            test_func = gen_nd_check_tests(j)
            setattr(ParetoTestsContainer, 'test_nd_check_%d' % j, test_func)
            fn = getattr(ParetoTestsContainer, 'test_nd_check_%d' % j)
            fn.__call__(self)

    def test_crowd_dist_pytest_wrapper(self):
        for j in indicies:
            test_func = gen_nd_check_tests(j)
            setattr(ParetoTestsContainer, 'test_crowd_dist_%d' % j, test_func)
            fn = getattr(ParetoTestsContainer, 'test_crowd_dist_%d' % j)
            fn.__call__(self)

def gen_nd_check_tests(j):
    def test(self):
        path = os.path.abspath(os.path.dirname(__file__))
        folder = os.path.join(path, "pareto_tools/nd_check_test")

        PF_set = np.loadtxt(os.path.join(folder, "test_%05d_PF_set" % j))
        stest = np.loadtxt(os.path.join(folder, "test_%05d_stest" % j))
        Jtest = np.loadtxt(os.path.join(folder, "test_%05d_Jtest" % j))

        matlab_PF = np.loadtxt(os.path.join(folder, "test_%05d_result" % j), delimiter=",")
        matlab_dom = np.loadtxt(os.path.join(folder, "test_%05d_dom" % j), delimiter=",")

        PF_set, dom = nd_check(PF_set, Jtest, stest)
        diff = np.abs(matlab_PF - PF_set)
        if not (diff < 5e-13).all():
            self.fail("Difference in results detected: " + str(np.max(diff)))

        self.assertEqual(dom, matlab_dom)
    return test


def gen_crowd_dist_tests(j):
    def test(self):
        #print("### TEST No " + str(j) + " of " + str(1000) + " ###", end="\r")
        path = os.path.abspath(os.path.dirname(__file__))
        folder = os.path.join(path, self.pareto_folder)

        filename = "test_file_%05d" % j
        testfile = os.path.join(folder, filename)

        resultfile = os.path.join(folder, "test_file_%05d_result" % j)

        m_arr = pd.read_table(testfile, header=None, sep='\s+').values
        matlab_result = np.loadtxt(resultfile)

        python_result = crowd_dist(m_arr)
        diff = np.abs(python_result - matlab_result)
        if not (diff < 5e-15).all():
            self.fail("Difference in results detected: " + str(np.max(diff)))
    return test


if __name__ == '__main__':
    for j in indicies:
        test_func = gen_nd_check_tests(j)
        setattr(ParetoTestsContainer, 'test_nd_check_%d' % j, test_func)

        test_cr_dist = gen_crowd_dist_tests(j)
        setattr(ParetoTestsContainer, 'test_crowd_dist_%d' % j, test_cr_dist)

    indicies = range(0)
    unittest.main()