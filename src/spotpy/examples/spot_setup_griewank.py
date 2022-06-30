'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This example implements the Griewank function into SPOT.	
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import spotpy

class spot_setup(object):
    def __init__(self, dim = 2):
        self.dim = dim
        self.params = []
        for i in range(self.dim):
            self.params.append(spotpy.parameter.Uniform(str(i), -20, 20, 2, 4.0))
        
    def parameters(self):
        return spotpy.parameter.generate(self.params)
                
  
    def simulation(self, vector):
        n = len(vector)
        fr = 4000
        s = 0
        p = 1
        for j in range(n): 
            s = s + vector[j]**2
        for j in range(n): 
            p = p * np.cos(vector[j] / np.sqrt(j+1))
        simulation = [s / fr - p + 1]
        return simulation     
       
    def evaluation(self):
        observations = [0]
        return observations
    
    def objectivefunction(self, simulation,evaluation):
        objectivefunction= -spotpy.objectivefunctions.rmse(evaluation = evaluation, simulation = simulation)
        return objectivefunction