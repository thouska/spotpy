# -*- coding: utf-8 -*-
'''
Copyright (c) 2017 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska, Benjamin Manns

This code shows how to use the likelihood framework and present all existing function.
'''

import numpy as np
try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy

import unittest

# We use all available likelihood functions. The pydoc of every function tells, if we can add a
# parameter `param` to the function which includes model parameter. The `param` must be None or a tuple with values
# and names. If `param` is None, the needed values are calculated by the function itself.


class TestLikelihood(unittest.TestCase):
    def setUp(self):
        self.data, self.comparedata = np.random.normal(1500, 2530, 20), np.random.normal(15, 25, 20)
        self.do_print = True

    def test_logLikelihood(self):
        l = spotpy.likelihoods.logLikelihood(self.data, self.comparedata)
        self.assertGreaterEqual(np.abs(l),900)
        self.assertEqual(type(np.float(l)),type(np.float(1)))
        if self.do_print:
            print("logLikelihood: " + str(l))

    def test_gaussianLikelihoodMeasErrorOut(self):
        l = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(self.data, self.comparedata)
        self.assertGreaterEqual(-100, l)
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("gaussianLikelihoodMeasErrorOut: " + str(l))

    def test_gaussianLikelihoodHomoHeteroDataError(self):
        l = spotpy.likelihoods.gaussianLikelihoodHomoHeteroDataError(self.data, self.comparedata)
        self.assertGreaterEqual(1,np.abs(l))
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("gaussianLikelihoodHomoHeteroDataError: " + str(l))

    def test_LikelihoodAR1NoC(self):
        l = spotpy.likelihoods.LikelihoodAR1NoC(self.data, self.comparedata, params=([0.98], ["likelihood_phi"]))
        self.assertNotEqual(None,l)
        if self.do_print:
            print("LikelihoodAR1NoC: " + str(l))

    def test_LikelihoodAR1WithC(self):
        l = spotpy.likelihoods.LikelihoodAR1WithC(self.data, self.comparedata)
        self.assertNotEqual(None, l)
        if self.do_print:
            print("LikelihoodAR1WithC: " + str(l))

    def test_generalizedLikelihoodFunction(self):
        size = 1000
        data, comparedata = np.random.normal(1500, 2530, size), np.random.normal(355, 25, size)

        l = spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata, params=
            ([np.random.uniform(-0.99, 1, 1), np.random.uniform(0.1, 10, 1), np.random.uniform(0, 1, 1), np.random.uniform(0, 1, 0),
            np.random.uniform(0, 0.99, 1), np.random.uniform(0, 100, 1)],
            ["likelihood_beta", "likelihood_xi", "likelihood_sigma0", "likelihood_sigma1", "likelihood_phi1", "likelihood_muh"]))

        self.assertNotEqual(None, l)
        self.assertGreaterEqual(-10000, l)
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("generalizedLikelihoodFunction: " + str(l))

    def test_LaplacianLikelihood(self):
        l = spotpy.likelihoods.LaplacianLikelihood(self.data, self.comparedata)
        self.assertNotEqual(None, l)
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("LaplacianLikelihood: " + str(l))

    def test_SkewedStudentLikelihoodHomoscedastic(self):
        l = spotpy.likelihoods.SkewedStudentLikelihoodHomoscedastic(self.data, self.comparedata)
        self.assertGreaterEqual(0.5, np.abs(l))
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("SkewedStudentLikelihoodHomoscedastic: " + str(l))

    def test_SkewedStudentLikelihoodHeteroscedastic(self):
        l = spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedastic(self.data, self.comparedata)
        if not np.isnan(l):
            self.assertGreaterEqual(-100, l)
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("SkewedStudentLikelihoodHeteroscedastic: " + str(l))

    def test_SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(self):
        l = spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(self.data, self.comparedata, params=(
            [np.random.uniform(2.01, 100, 1)[0], np.random.uniform(0.01, 100, 1)[0], np.random.uniform(-.99, .99, 1)[0]],
            ["likelihood_nu", "likelihood_kappa", "likelihood_phi"]))

        self.assertNotEqual(None, l)
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("SkewedStudentLikelihoodHeteroscedasticAdvancedARModel: " + str(l))

    def test_NoisyABCGaussianLikelihood(self):
        l = spotpy.likelihoods.NoisyABCGaussianLikelihood(self.data, self.comparedata)
        self.assertNotEqual(None, l)
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("NoisyABCGaussianLikelihood: " + str(l))

    def test_ABCBoxcarLikelihood(self):
        l = spotpy.likelihoods.ABCBoxcarLikelihood(self.data, self.comparedata)
        self.assertNotEqual(None, l)
        self.assertNotEqual(np.nan, l)
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("ABCBoxcarLikelihood: " + str(l))

    def test_LimitsOfAcceptability(self):
        l = spotpy.likelihoods.LimitsOfAcceptability(self.data, self.comparedata)
        self.assertGreaterEqual(20, l)
        self.assertNotEqual(None, l)
        self.assertEqual(type(np.int(l)), type(int(1)))
        if self.do_print:
            print("LimitsOfAcceptability: " + str(l))

    def test_InverseErrorVarianceShapingFactor(self):
        l = spotpy.likelihoods.InverseErrorVarianceShapingFactor(self.data, self.comparedata)
        self.assertGreaterEqual(-20000, l)
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("inverseErrorVarianceShapingFactor: " + str(l))

    def test_ExponentialTransformErrVarShapingFactor(self):
        l = spotpy.likelihoods.ExponentialTransformErrVarShapingFactor(self.data, self.comparedata)
        self.assertGreaterEqual(-30000, l)
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("inverseErrorVarianceShapingFactor: " + str(l))

    def test_NashSutcliffeEfficiencyShapingFactor(self):
        l = spotpy.likelihoods.NashSutcliffeEfficiencyShapingFactor(self.data, self.comparedata)
        self.assertNotEqual(None, l)
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("NashSutcliffeEfficiencyShapingFactor: " + str(l))

    def test_sumOfAbsoluteErrorResiduals(self):
        l = spotpy.likelihoods.sumOfAbsoluteErrorResiduals(self.data, self.comparedata)
        self.assertGreaterEqual(2,np.abs(np.abs(l)-10))
        self.assertEqual(type(np.float(l)), type(np.float(1)))
        if self.do_print:
            print("sumOfAbsoluteErrorResiduals: " + str(l))

if __name__ == '__main__':
    while True:
        unittest.main()