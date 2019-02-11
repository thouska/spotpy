import numpy as np
import spotpy

from spotpy.unittests.test_dds import FixedRandomizer
from spotpy.unittests.test_dds import FixedRandomizerEndOfDataException

def ZDT1(x):
    """
    This test function is used by Deb et al. 2002 IEEE to test NSGAII
    performance. There are 30 decision variables which are in [0,1].
    :param x:
    :return: Two Value Tuple
    """
    a = x[0] # objective 1 value
    g = 0
    for i in range(1,30):
        g = g + x[i]
    g = 1 + 9 * g / 29
    b = g * (1 - (x[0] / g) ** 0.5) # objective 2 value
    return a,b


class spot_setup(object):
    def __init__(self):
        self.params = []
        for i in range(30):
            self.params.append(spotpy.parameter.Uniform(0, 1, 0, 0, 0, 1,doc="param no " + str(i+1)))

    def parameters(self):

        a = spotpy.parameter.generate(self.params)

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

    def objectivefunction(self, simulation, evaluation):
        # https://deap.readthedocs.io/en/master/api/benchmarks.html  # deap.benchmarks.zdt1
        # ZDT1 function
        if len(simulation) != 30:
            raise Exception("simulation must have length 30")
        return ZDT1(simulation)

spot_setup = spot_setup()

sampler = spotpy.algorithms.padds(spot_setup, dbname='padds_hymod', dbformat='csv', alt_objfun=None)
fr = FixedRandomizer()
sampler._set_np_random(fr)
res = sampler.sample(458,trials=1)
print(res)
