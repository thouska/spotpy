import spotpy
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import rmse
import numpy as np


class spot_setup(object):

    def __init__(self):
        self.params = [Uniform(str(j),-2, 2, 1.5, 3.0, -2, 2, doc=str(j)+' value of Rosenbrock function')
                       for j in range(10)]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        x = np.array(vector)
        simulations = [sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)]
        return simulations

    def evaluation(self):
        observations = [0]
        return observations

    def objectivefunction(self, simulation, evaluation):
        objectivefunction = -rmse(evaluation=evaluation, simulation=simulation)
