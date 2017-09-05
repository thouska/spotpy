# -*- coding: utf-8 -*-
'''
Copyright (c) 2017 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska, Benjamin Manns

This code shows how to use the likelihood framework and present all existing function.
'''

import numpy as np
import spotpy

# First we use all available likelihood functions just alone. The pydoc of every function tells, if we can add a
# parameter `param` to the function which includes model parameter. The `param` musst be None or a tuple with values
# and names. If `param` is None, the needed values are calculated by the function itself.

data, comparedata = np.random.normal(150, 250, 20), np.random.normal(15, 25, 20)


l = spotpy.likelihoods.logLikelihood(data, comparedata)
print("logLikelihood: " + str(l))

l = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(data, comparedata)
print("gaussianLikelihoodMeasErrorOut: " + str(l))

l = spotpy.likelihoods.gaussianLikelihoodHomoHeteroDataError(data, comparedata)
print("gaussianLikelihoodHomoHeteroDataError: " + str(l))

# Here are examples where functions get `params`
l = spotpy.likelihoods.LikelihoodAR1NoC(data, comparedata,params=([0.98],["likelihood_phi"]))
print("LikelihoodAR1NoC: " + str(l))

l = spotpy.likelihoods.LikelihoodAR1WithC(data, comparedata)
print("LikelihoodAR1WithC: " + str(l))

l = spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata,params=
                ([np.random.uniform(-0.99,1,1),np.random.uniform(0.1,10,1),np.random.uniform(0,1,1),np.random.uniform(0, 1,0),np.random.uniform(0, 0.99, 1),np.random.uniform(0, 100, 1)],
                                                                               ["likelihood_beta","likelihood_xsi","likelihood_sigma0","likelihood_sigma1","likelihood_phi1","likelihood_muh"]))
print("generalizedLikelihoodFunction: " + str(l))

l = spotpy.likelihoods.LaplacianLikelihood(data, comparedata)
print("LaplacianLikelihood: " + str(l))

l = spotpy.likelihoods.SkewedStudentLikelihoodHomoscedastic(data, comparedata)
print("SkewedStudentLikelihoodHomoscedastic: " + str(l))

l = spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedastic(data, comparedata)
print("SkewedStudentLikelihoodHeteroscedastic: " + str(l))

l = spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(data, comparedata,params=([np.random.uniform(2.01,100,1),np.random.uniform(0.01,100,1),np.random.uniform(-.99, .99,1)],
                                                                                                        ["likelihood_nu","likelihood_kappa","likelihood_phi"]))

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

# We also can use the likelihood functions in an algorithmus. We will need a setup class like this


class spot_setup(object):
    def __init__(self):
        self.params = [spotpy.parameter.Uniform('x', -10, 10, 1.5, 3.0, -10, 10),
                       spotpy.parameter.Uniform('y', -170, 140, 1.55, 3.01, -154, 107),

                       # Some likelihood function need additional parameter, look them up in the documentation
                       spotpy.parameter.Uniform('likelihood_nu', 2.01, 100, 2.01, 100, 2.01, 100),
                       spotpy.parameter.Uniform('likelihood_kappa', 0.01, 100, 0.01, 100, 0.01, 100),
                       spotpy.parameter.Uniform('likelihood_phi', -.99, .99, -.99, .99, -0.5, .4),

                       spotpy.parameter.Uniform('likelihood_beta', -.99, .99, -.99, .99, -0.5, .4),
                       spotpy.parameter.Uniform('likelihood_xsi', 0.11, 10, -.99, .99, -0.5, .4),
                       spotpy.parameter.Uniform('likelihood_sigma0', 0, 1, -.99, .99, -0.5, .4),
                       spotpy.parameter.Uniform('likelihood_sigma1', 0, 1, -.99, .99, -0.5, .4),
                       spotpy.parameter.Uniform('likelihood_phi1', 0, .99, -.99, .99, -0.5, .4),
                       spotpy.parameter.Uniform('likelihood_muh', 0, 100, -.99, .99, -0.5, .4)

                       ]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        x = np.random.randint(-100, 100, size=100)
        # simulations= [sum(100.0 * (x[1:] - x[:-1] **2.0) **2.0 + (1 - x[:-1]) **2.0)]
        simulations = x

        return simulations

    def evaluation(self):
        observations = np.random.randint(-100, 100, size=100)
        return observations

    def objectivefunction(self, simulation=simulation, evaluation=evaluation, params=None):
        # Some functions do not nee a `param` attribute, you will see that in the documentation or if an error occur.
        objectivefunction = spotpy.likelihoods.generalizedLikelihoodFunction(evaluation, simulation,params=params)

        return objectivefunction



# And now we can start an algorithm to find the best possible data


results=[]
spot_setup=spot_setup()
rep=5000

sampler=spotpy.algorithms.mc(spot_setup,    dbname='RosenMC',    dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())
print(results)