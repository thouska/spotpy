import spotpy
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import rmse
import numpy as np


def ackley10(vector):
    length = len(vector)
    sum1 = 0
    sum2 = 0
    for i in range(length):
        sum1 = sum1 + vector[i] ** 2
        sum2 = sum2 + np.cos(2 * np.pi * vector[i])
    return -1*(-20 * np.exp(-0.2 * (sum1 / length) ** 0.5) - np.exp(sum2 / length))


def griewank10(vector):
    sum1 = 0
    term2 = 1
    term3 = 1

    for i in range(len(vector)):
        sum1 = sum1 + (vector[i] ** 2) / 4000
        term2 = term2 * np.cos(vector[i] / (i + 1) ** 0.5)

    return -1*(sum1 - term2 + term3)


class spot_setup(object):
    """
        Setup for a simple example to run DDS Algorithm
    """

    def __init__(self):
        self.params = None
        self.objfunc = None

    def _objfunc_switcher(self, name):
        """
        Set new parameter and objective function while setup is instanced in a test case
        :param name: function name which overwrites initial objective function
        :return:
        """

        if name == "ackley":
            self.objfunc = ackley10
            self.params = [Uniform(str(j), -2, 2, 1.5, 3.0, -2, 2, doc=str(j) + ' value of Rosenbrock function')
                           for j in range(10)]
        elif name == "griewank":
            self.objfunc = griewank10
            self.params = [Uniform('d' + str(j), -500, 700, 1.5, 3.0, -500, 700,
                                   doc=str(j) + 'distinc parameter within a boundary', as_int=True)
                           for j in range(2)] + [Uniform('c' + str(j), -500, 700, 1.5, 3.0, -500, 700,
                                                         doc=str(j) + 'continuous parameter within a boundary')
                                                 for j in range(8)]
        elif name == "cmf_style":
            self.objfunc = ackley10
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

    def parameters(self):
        if self.params is None:
            self.params = [
                Uniform("0", -10, 10, 1.5, 3.0, -10, 10, doc='x value of Rosenbrock function'),
                Uniform("1", -10, 10, 1.5, 3.0, -10, 10, doc='y value of Rosenbrock function'),
                Uniform("z", -10, 10, 1.5, 3.0, -10, 10, doc='z value of Rosenbrock function')]
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        x = np.array(vector)
        # simulations = [sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)]
        simulations = x * np.random.rand(len(vector))
        #simulations = x * np.sum(vector)
        return simulations

    def evaluation(self):
        # observations = [0]
        observations = [2, 3, 4]
        return observations

    def objectivefunction(self, simulation, evaluation, params):

        if self.objfunc is None:
            return -1*rmse(evaluation, simulation)
        else:
            pars, names = params
            return self.objfunc(pars)
