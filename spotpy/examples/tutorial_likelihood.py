# -*- coding: utf-8 -*-
'''
Copyright (c) 2017 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska, Benjamin Manns

This code shows how to use the likelihood framework and present all existing function.
'''

import numpy as np
import spotpy


data, comparedata = np.random.uniform(150, 250, 20), np.random.uniform(15, 25, 20)
data, comparedata = np.random.normal(150, 250, 20), np.random.normal(15, 25, 20)


l = spotpy.likelihoods.logLikelihood(data, comparedata)
print("logLikelihood: " + str(l))

l = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(data, comparedata)
print("gaussianLikelihoodMeasErrorOut: " + str(l))

l = spotpy.likelihoods.gaussianLikelihoodHomoHeteroDataError(data, comparedata)
print("gaussianLikelihoodHomoHeteroDataError: " + str(l))

l = spotpy.likelihoods.LikelihoodAR1NoC(data, comparedata)
print("LikelihoodAR1NoC: " + str(l))

l = spotpy.likelihoods.LikelihoodAR1WithC(data, comparedata)
print("LikelihoodAR1WithC: " + str(l))

l = spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata)
print("generalizedLikelihoodFunction: " + str(l))

l = spotpy.likelihoods.LaplacianLikelihood(data, comparedata)
print("LaplacianLikelihood: " + str(l))

l = spotpy.likelihoods.SkewedStudentLikelihoodHomoscedastic(data, comparedata)
print("SkewedStudentLikelihoodHomoscedastic: " + str(l))

l = spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedastic(data, comparedata)
print("SkewedStudentLikelihoodHeteroscedastic: " + str(l))

l = spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(data, comparedata)
print("SkewedStudentLikelihoodHeteroscedasticAdvancedARModel: " + str(l))

l = spotpy.likelihoods.NoisyABCGaussianLikelihood(data, comparedata)
print("NoisyABCGaussianLikelihood: " + str(l))

l = spotpy.likelihoods.ABCBoxcarLikelihood(data, comparedata)
print("ABCBoxcarLikelihood: " + str(l))

l = spotpy.likelihoods.LimitsOfAcceptability(data, comparedata)
print("LimitsOfAcceptability: " + str(l))

l = spotpy.likelihoods.InverseErrorVarianceShapingFactor(data, comparedata)
print("inverseErrorVarianceShapingFactor: " + str(l))

l = spotpy.likelihoods.ExponentialTransformErrVarShapingFactor(data, comparedata)
print("inverseErrorVarianceShapingFactor: " + str(l))

l = spotpy.likelihoods.NashSutcliffeEfficiencyShapingFactor(data, comparedata)
print("NashSutcliffeEfficiencyShapingFactor: " + str(l))

l = spotpy.likelihoods.sumOfAbsoluteErrorResiduals(data, comparedata)
print("sumOfAbsoluteErrorResiduals: " + str(l))

