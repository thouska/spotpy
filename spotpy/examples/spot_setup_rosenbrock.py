'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska
This example implements the Rosenbrock function into SPOT.  
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import spotpy

        
class spot_setup(object):
    def __init__(self):
        self.params = [spotpy.parameter.Uniform('x', -10, 10, 1.5, 3.0, -10, 10),
                       spotpy.parameter.Uniform('y', -10, 10, 1.5, 3.0, -10, 10)
                       ]
    def parameters(self):
        return spotpy.parameter.generate(self.params)
        
    def simulation(self,vector):      
        x=np.array(vector)
        simulations= [sum(100.0 * (x[1:] - x[:-1] **2.0) **2.0 + (1 - x[:-1]) **2.0)]
        return simulations
        
    def evaluation(self):
        observations = [0]
        return observations
    
    def objectivefunction(self, simulation = simulation, evaluation = evaluation):
        objectivefunction = -spotpy.objectivefunctions.rmse(evaluation = evaluation, simulation = simulation)      
        return objectivefunction