# -*- coding: utf-8 -*-
'''
Copyright (c) 2017 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska, Benjamin Manns

This code shows how to use the likelihood framework and present all existing function.
'''

import numpy as np
from spotpy.likelihood import *


data, comparedata = np.random.uniform(150, 250, 20), np.random.uniform(15, 25, 20)
data, comparedata = np.random.normal(150, 250, 20), np.random.normal(15, 25, 20)


l = logLikelihood(data, comparedata)
print("logLikelihood: " + str(l))

l = gaussianLikelihoodMeasErrorOut(data, comparedata)
print("gaussianLikelihoodMeasErrorOut: " + str(l))

l = gaussianLikelihoodHomoHeteroDataError(data, comparedata)
print("gaussianLikelihoodHomoHeteroDataError: " + str(l))

l = LikelihoodAR1NoC(data, comparedata)
print("LikelihoodAR1NoC: " + str(l))

l = LikelihoodAR1WithC(data, comparedata)
print("LikelihoodAR1WithC: " + str(l))

l = generalizedLikelihoodFunction(data, comparedata)
print("generalizedLikelihoodFunction: " + str(l))

l = LaplacianLikelihood(data, comparedata)
print("LaplacianLikelihood: " + str(l))

l = SkewedStudentLikelihoodHomoscedastic(data, comparedata)
print("SkewedStudentLikelihoodHomoscedastic: " + str(l))

l = SkewedStudentLikelihoodHeteroscedastic(data, comparedata)
print("SkewedStudentLikelihoodHeteroscedastic: " + str(l))

l = SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(data, comparedata)
print("SkewedStudentLikelihoodHeteroscedasticAdvancedARModel: " + str(l))

l = NoisyABCGaussianLikelihood(data, comparedata)
print("NoisyABCGaussianLikelihood: " + str(l))

l = ABCBoxcarLikelihood(data, comparedata)
print("ABCBoxcarLikelihood: " + str(l))

l = LimitsOfAcceptability(data, comparedata)
print("LimitsOfAcceptability: " + str(l))

l = InverseErrorVarianceShapingFactor(data, comparedata)
print("inverseErrorVarianceShapingFactor: " + str(l))

l = ExponentialTransformErrVarShapingFactor(data, comparedata)
print("inverseErrorVarianceShapingFactor: " + str(l))

l = NashSutcliffeEfficiencyShapingFactor(data, comparedata)
print("NashSutcliffeEfficiencyShapingFactor: " + str(l))

l = sumOfAbsoluteErrorResiduals(data, comparedata)
print("sumOfAbsoluteErrorResiduals: " + str(l))

