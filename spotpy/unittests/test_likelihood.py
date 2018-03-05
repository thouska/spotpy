# -*- coding: utf-8 -*-
"""
Copyright (c) 2017 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska, Benjamin Manns

This code tests the likelihood framework and present all existing function.
"""

import numpy as np

try:
    import spotpy
except ImportError:
    import sys

    sys.path.append(".")
    import spotpy

import unittest
from spotpy.likelihoods import LikelihoodError


# We use all available likelihood functions. The pydoc of every function tells, if we can add a
# parameter `param` to the function which includes model parameter. The `param` must be None or a tuple with values
# and names. If `param` is None, the needed values are calculated by the function itself.


class TestLikelihood(unittest.TestCase):
    def setUp(self):
        np.random.seed(12)
        self.normal_data, self.normal_comparedata = np.random.normal(1500, 2530, 20), np.random.normal(15, 25, 20)
        self.binom_data, self.binom_comparedata = np.random.binomial(20, 0.1, 20), np.random.binomial(20, 0.1, 20)
        self.do_print = True

    def test_logLikelihood(self):
        l_normal = spotpy.likelihoods.logLikelihood(self.normal_data, self.normal_comparedata)
        self.assertGreaterEqual(np.abs(l_normal), 900)
        self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
        if self.do_print:
            print("logLikelihood: " + str(l_normal))

        l_binom = spotpy.likelihoods.logLikelihood(self.binom_data, self.binom_comparedata)
        self.assertGreaterEqual(np.abs(l_binom), 900)
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("logLikelihood: " + str(l_binom))

    def test_gaussianLikelihoodMeasErrorOut(self):
        l_normal = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(self.normal_data, self.normal_comparedata)
        self.assertGreaterEqual(-40, l_normal)
        self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
        if self.do_print:
            print("gaussianLikelihoodMeasErrorOut: " + str(l_normal))

        l_binom = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(self.binom_data, self.binom_comparedata)
        self.assertGreaterEqual(-40, l_binom)
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("gaussianLikelihoodMeasErrorOut: " + str(l_binom))

    def test_gaussianLikelihoodHomoHeteroDataError(self):
        l_normal = spotpy.likelihoods.gaussianLikelihoodHomoHeteroDataError(self.normal_data, self.normal_comparedata)
        self.assertGreaterEqual(5, np.abs(l_normal))
        self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
        if self.do_print:
            print("gaussianLikelihoodHomoHeteroDataError: " + str(l_normal))

        l_binom = spotpy.likelihoods.gaussianLikelihoodHomoHeteroDataError(self.binom_data, self.binom_comparedata)
        self.assertGreaterEqual(10, np.abs(l_binom))
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("gaussianLikelihoodHomoHeteroDataError: " + str(l_binom))

    def test_LikelihoodAR1NoC(self):
        l_list = []

        l_list.append(spotpy.likelihoods.LikelihoodAR1NoC(self.normal_data, self.normal_comparedata,
                                                          params=([0.98], ["likelihood_phi"])))

        try:
            l_list.append(spotpy.likelihoods.LikelihoodAR1NoC(self.normal_data, self.normal_comparedata,
                                                              params=([], [])))
        except LikelihoodError as e:
            print("LikelihoodError occurred: " + str(e))

        l_list.append(spotpy.likelihoods.LikelihoodAR1NoC(self.normal_data, self.normal_comparedata,
                                                          params=([1.1], ["likelihood_phi"])))

        l_list.append(spotpy.likelihoods.LikelihoodAR1NoC(self.binom_data, self.binom_data))

        for l in l_list:
            self.assertNotEqual(None, l)
            if self.do_print:
                print("LikelihoodAR1NoC: " + str(l))

    def test_LikelihoodAR1WithC(self):
        l_normal_list = []
        try:
            l_normal_list.append(spotpy.likelihoods.LikelihoodAR1WithC(self.normal_data, self.normal_comparedata,
                                                                       params=([], [])))
        except LikelihoodError as e:
            print("Likelihood Error occurred " + str(e))

        l_normal_list.append(spotpy.likelihoods.LikelihoodAR1WithC(self.normal_data, self.normal_comparedata,
                                                                   params=([0.98], ["likelihood_phi"])))
        l_normal_list.append(spotpy.likelihoods.LikelihoodAR1WithC(self.normal_data, self.normal_comparedata,
                                                                   params=([1.1], ["likelihood_phi"])))

        l_normal_list.append(spotpy.likelihoods.LikelihoodAR1WithC(self.binom_data, self.binom_comparedata))

        for l_normal in l_normal_list:
            self.assertNotEqual(None, l_normal)
            if self.do_print:
                print("LikelihoodAR1WithC: " + str(l_normal))

    def test_generalizedLikelihoodFunction(self):
        size = 1000
        data, comparedata = np.random.normal(1500, 2530, size), np.random.normal(355, 25, size)

        param_list = ["likelihood_beta", "likelihood_xi", "likelihood_sigma0", "likelihood_sigma1", "likelihood_phi1",
                      "likelihood_muh"]
        l_normal_list = []

        l_normal_list.append(spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata, params=
        ([-0.09, 1, 0.5, 0.567, 0.98, 57.32], param_list)))

        try:
            l_normal_list.append(spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata, params=
            ([], [])))
        except LikelihoodError as e:
            print("Likelihood Error occurred " + str(e))

        l_normal_list.append(spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata, params=
        ([2, 1, 0.5, 0.567, 0.98, 57.32], param_list)))
        l_normal_list.append(spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata, params=
        ([-0.09, 11, 0.5, 0.567, 0.98, 57.32], param_list)))
        l_normal_list.append(spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata, params=
        ([-0.09, 1, 1.5, 0.567, 0.98, 57.32], param_list)))
        l_normal_list.append(spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata, params=
        ([-0.09, 1, 0.5, 1.567, 0.98, 57.32], param_list)))
        l_normal_list.append(spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata, params=
        ([-0.09, 1, 0.5, 0.567, 2.98, 57.32], param_list)))
        l_normal_list.append(spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata, params=
        ([-0.09, 1, 0.5, 0.567, 0.98, 101], param_list)))
        l_normal_list.append(spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata, params=
        ([-0.09, 0.0, 0.5, 0.567, 0.98, 101], param_list)))

        for l_normal in l_normal_list:

            self.assertNotEqual(None, l_normal)
            self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
            if self.do_print:
                print("generalizedLikelihoodFunction: " + str(l_normal))

        l_binom = spotpy.likelihoods.generalizedLikelihoodFunction(self.binom_data, self.binom_comparedata)

        self.assertNotEqual(None, l_binom)
        self.assertGreaterEqual(-10000, l_binom)
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("generalizedLikelihoodFunction: " + str(l_binom))

    def test_LaplacianLikelihood(self):
        l_normal = spotpy.likelihoods.LaplacianLikelihood(self.normal_data, self.normal_comparedata)
        self.assertNotEqual(None, l_normal)
        self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
        if self.do_print:
            print("LaplacianLikelihood: " + str(l_normal))

        l_binom = spotpy.likelihoods.LaplacianLikelihood(self.binom_data, self.binom_comparedata)
        self.assertNotEqual(None, l_normal)
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("LaplacianLikelihood: " + str(l_binom))

    def test_SkewedStudentLikelihoodHomoscedastic(self):
        l_normal = spotpy.likelihoods.SkewedStudentLikelihoodHomoscedastic(self.normal_data, self.normal_comparedata)
        self.assertGreaterEqual(12, np.abs(l_normal))
        self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
        if self.do_print:
            print("SkewedStudentLikelihoodHomoscedastic: " + str(l_normal))

        l_binom = spotpy.likelihoods.SkewedStudentLikelihoodHomoscedastic(self.binom_data, self.binom_comparedata)
        self.assertGreaterEqual(17, np.abs(l_binom))
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("SkewedStudentLikelihoodHomoscedastic: " + str(l_binom))

    def test_SkewedStudentLikelihoodHeteroscedastic(self):
        l_normal_list = []
        paramDependencies = ["likelihood_nu", "likelihood_kappa", "likelihood_phi"]
        l_normal_list.append(
            spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedastic(self.normal_data, self.normal_comparedata,
                                                                      params=([2.4, 0.15, 0.87], paramDependencies)))
        try:
            l_normal_list.append(
                spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedastic(self.normal_data, self.normal_comparedata,
                                                                          params=([], [])))
        except LikelihoodError as e:
            print("An error occurred: " + str(e))

        l_normal_list.append(
            spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedastic(self.normal_data, self.normal_comparedata,
                                                                      params=([1, 0.15, 1.87], paramDependencies)))

        l_normal_list.append(
            spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedastic(self.normal_data, self.normal_comparedata,
                                                                      params=([1, 0.15, 0.87], paramDependencies)))

        l_normal_list.append(
            spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedastic(self.normal_data, self.normal_comparedata,
                                                                      params=([1, -0.15, 0.87], paramDependencies)))
        for l_normal in l_normal_list:
            if not np.isnan(l_normal):
                self.assertGreaterEqual(-100, l_normal)
            self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
            if self.do_print:
                print("SkewedStudentLikelihoodHeteroscedastic: " + str(l_normal))

        l_binom = spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedastic(self.binom_data, self.binom_comparedata)
        if not np.isnan(l_binom):
            self.assertGreaterEqual(-100, l_binom)
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("SkewedStudentLikelihoodHeteroscedastic: " + str(l_binom))

    def test_SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(self):
        l_normal_list = []
        params = ["likelihood_nu", "likelihood_kappa", "likelihood_phi"]

        l_normal_list.append(spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(
            self.normal_data, self.normal_comparedata, params=([4, 43, 0.4], params)))

        try:
            l_normal_list.append(spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(
                self.normal_data, self.normal_comparedata, params=([], [])))
        except LikelihoodError as e:
            print("Likelihood Error occurred " + str(e))

        l_normal_list.append(spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(
            self.normal_data, self.normal_comparedata, params=([4, 43, 2.4], params)))
        l_normal_list.append(spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(
            self.normal_data, self.normal_comparedata, params=([1, 43, 0.4], params)))
        l_normal_list.append(spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(
            self.normal_data, self.normal_comparedata, params=([4, -3, 0.4], params)))

        for l_normal in l_normal_list:
            self.assertNotEqual(None, l_normal)
            self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
            if self.do_print:
                print("SkewedStudentLikelihoodHeteroscedasticAdvancedARModel: " + str(l_normal))

        l_binom = spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(
            self.normal_data, self.normal_comparedata)

        self.assertNotEqual(None, l_binom)
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("SkewedStudentLikelihoodHeteroscedasticAdvancedARModel: " + str(l_binom))

    def test_NoisyABCGaussianLikelihood(self):
        l_normal = spotpy.likelihoods.NoisyABCGaussianLikelihood(self.normal_data, self.normal_comparedata)
        self.assertNotEqual(None, l_normal)
        self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
        if self.do_print:
            print("NoisyABCGaussianLikelihood: " + str(l_normal))

        l_binom = spotpy.likelihoods.NoisyABCGaussianLikelihood(self.binom_data, self.binom_data,
                                                                measerror=[0.0])
        self.assertNotEqual(None, l_binom)
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("NoisyABCGaussianLikelihood: " + str(l_binom))

    def test_ABCBoxcarLikelihood(self):
        l_normal = spotpy.likelihoods.ABCBoxcarLikelihood(self.normal_data, self.normal_comparedata)
        self.assertNotEqual(None, l_normal)
        self.assertNotEqual(np.nan, l_normal)
        self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
        if self.do_print:
            print("ABCBoxcarLikelihood: " + str(l_normal))

        l_binom = spotpy.likelihoods.ABCBoxcarLikelihood(self.binom_data, self.binom_comparedata)
        self.assertNotEqual(None, l_binom)
        self.assertNotEqual(np.nan, l_binom)
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("ABCBoxcarLikelihood: " + str(l_binom))

    def test_LimitsOfAcceptability(self):
        l_normal = spotpy.likelihoods.LimitsOfAcceptability(self.normal_data, self.normal_comparedata)
        self.assertEqual(12, l_normal)
        self.assertNotEqual(None, l_normal)
        self.assertEqual(type(np.int(l_normal)), type(int(1)))
        if self.do_print:
            print("LimitsOfAcceptability: " + str(l_normal))

        l_binom = spotpy.likelihoods.LimitsOfAcceptability(self.binom_data, self.binom_comparedata)
        self.assertEqual(5, l_binom)
        self.assertNotEqual(None, l_binom)
        self.assertEqual(type(np.int(l_binom)), type(int(1)))
        if self.do_print:
            print("LimitsOfAcceptability: " + str(l_binom))

    def test_InverseErrorVarianceShapingFactor(self):
        l_normal = spotpy.likelihoods.InverseErrorVarianceShapingFactor(self.normal_data, self.normal_comparedata)
        self.assertGreaterEqual(-10, l_normal)
        self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
        if self.do_print:
            print("inverseErrorVarianceShapingFactor: " + str(l_normal))

        l_binom = spotpy.likelihoods.InverseErrorVarianceShapingFactor(self.binom_data, self.binom_comparedata)
        self.assertGreaterEqual(-10, l_binom)
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("inverseErrorVarianceShapingFactor: " + str(l_binom))

    def test_ExponentialTransformErrVarShapingFactor(self):
        l_binom = spotpy.likelihoods.ExponentialTransformErrVarShapingFactor(self.binom_data, self.binom_comparedata)
        self.assertGreaterEqual(-30, l_binom)
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("inverseErrorVarianceShapingFactor: " + str(l_binom))

        l_gauss = spotpy.likelihoods.ExponentialTransformErrVarShapingFactor(self.normal_data, self.normal_comparedata)
        self.assertGreaterEqual(-30, l_gauss)
        self.assertEqual(type(np.float(l_gauss)), type(np.float(1)))
        if self.do_print:
            print("inverseErrorVarianceShapingFactor: " + str(l_gauss))

    def test_NashSutcliffeEfficiencyShapingFactor(self):
        l_normal_list = []
        l_normal_list.append(spotpy.likelihoods.NashSutcliffeEfficiencyShapingFactor(self.normal_data,
                                                                                     self.normal_comparedata))

        l_normal_list.append(spotpy.likelihoods.NashSutcliffeEfficiencyShapingFactor(self.normal_data,
                                                                                     self.normal_data))

        l_normal_list.append(spotpy.likelihoods.NashSutcliffeEfficiencyShapingFactor(self.binom_data,
                                                                                     self.binom_comparedata))

        try:
            l_normal_list.append(spotpy.likelihoods.NashSutcliffeEfficiencyShapingFactor([],
                                                                                         []))
        except LikelihoodError as e:
            print("Likelihood Error occurred: " + str(e))
        try:
            l_normal_list.append(spotpy.likelihoods.NashSutcliffeEfficiencyShapingFactor([1],
                                                                                         []))
        except LikelihoodError as e:
            print("Likelihood Error occurred " + str(e))

        for l_normal in l_normal_list:
            self.assertNotEqual(None, l_normal)
            self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
            if self.do_print:
                print("NashSutcliffeEfficiencyShapingFactor: " + str(l_normal))

    def test_sumOfAbsoluteErrorResiduals(self):
        l_normal = spotpy.likelihoods.sumOfAbsoluteErrorResiduals(self.normal_data, self.normal_comparedata)
        self.assertGreaterEqual(7, np.abs(np.abs(l_normal) - 10))
        self.assertEqual(type(np.float(l_normal)), type(np.float(1)))
        if self.do_print:
            print("sumOfAbsoluteErrorResiduals: " + str(l_normal))

        l_binom = spotpy.likelihoods.sumOfAbsoluteErrorResiduals(self.binom_data, self.binom_comparedata)
        self.assertGreaterEqual(7, np.abs(np.abs(l_binom) - 10))
        self.assertEqual(type(np.float(l_binom)), type(np.float(1)))
        if self.do_print:
            print("sumOfAbsoluteErrorResiduals: " + str(l_binom))


if __name__ == '__main__':
    unittest.main()
