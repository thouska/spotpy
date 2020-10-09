# -*- coding: utf-8 -*-
import numpy as np
import sys
from spotpy.parameter import Uniform
try:
    import spotpy
except ModuleNotFoundError:
    sys.path.append(".")
    import spotpy


def ZDT1(x):
    """
    Zitzler–Deb–Thiele's function number 1. Is used to benchmark or test algorithms, see also
    https://en.wikipedia.org/wiki/Test_functions_for_optimization and Deb et al. 2002 IEEE.


    ## The schematic tradoff looks like this
    # /\
    #  |
    #1 .
    #  |
    #  |
    #  | .
    #  |
    #  |   .
    #  |
    #  |      .
    #  |
    #  |           .
    #  |
    #  |                 .
    #  |                        .
    #  |                                 .
    #  |------------------------------------------.------>
    #                                             1

    ZDT1 needs 30 parameters, which are in [0,1].
    :param x:
    :return: Two Value Tuple
    """
    a = x[0] # objective 1 value
    g = 0
    for i in range(1,30):
        g = g + x[i]
    g = 1 + 9 * g / 29
    b = g * (1 - (x[0] / g) ** 0.5) # objective 2 value
    return np.array([a,b])


class padds_spot_setup(object):
    def __init__(self, default=True):
        self.params = []
        if default:
            for i in range(30):
                self.params.append(spotpy.parameter.Uniform(str(i+1), 0, 1, 0, 0, 0, 1,doc="param no " + str(i+1)))
        else:
            self.params = [Uniform(.5, 5., optguess=1.5, doc='saturated depth at beginning'),
                           Uniform(.001, .8, optguess=.1, doc='porosity of matrix [m3 Pores / m3 Soil]'),
                           Uniform(1., 240., optguess=10.,
                                   doc='ssaturated conductivity of macropores [m/day]'),
                           Uniform(.0001, .5, optguess=.05, doc='macropore fraction [m3/m3]'),
                           Uniform(.005, 1., optguess=.05,
                                   doc='mean distance between the macropores [m]'),
                           Uniform(0., 1., optguess=0.,
                                   doc='water content when matric potential pointing towards -infinity'),
                           Uniform(.5, 1., optguess=.99,
                                   doc='wetness above which the parabolic extrapolation is used instead of VGM'),
                           Uniform(0., 50, optguess=.1,
                                   doc='exchange rate [1/day] for macropore-matrix-exchange')]
            for i in range(8,30):
                self.params.append(Uniform(str(i+1), 0, 1, 0, 0, 0, 1,doc="param no " + str(i+1)))

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        firstSum = 0.0
        secondSum = 0.0
        for c in range(len(vector)):
            firstSum += c**2.0
            secondSum += np.cos(2.0*np.pi*vector[c])
            n = float(len(vector))
        return [-20.0*np.exp(-0.2*np.sqrt(firstSum/n)) - np.exp(secondSum/n) + 20 + np.e]

    def evaluation(self):
        observations = [0]
        return observations

    def objectivefunction(self, simulation, evaluation, params):
        para, names = params
        if len(para) != 30:
            raise Exception("params must have length 30")
        return ZDT1(para)

spot_setup = padds_spot_setup()

sampler = spotpy.algorithms.padds(spot_setup, dbname='padds_hymod', dbformat='csv')
res = sampler.sample(2000,trials=1)
