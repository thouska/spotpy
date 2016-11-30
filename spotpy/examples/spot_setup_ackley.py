'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This example implements the Ackley function into SPOT.  
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import spotpy

class spot_setup(object):
    def __init__(self,dim=30):
        self.dim=dim
        self.params = []
        for i in range(self.dim):
            self.params.append(spotpy.parameter.Uniform(str(i),-32.768,32.768,2.5,-20.0))
        
    def parameters(self):
        return spotpy.parameter.generate(self.params)
                
  
    def simulation(self, vector):
        firstSum = 0.0
        secondSum = 0.0
        for c in range(len(vector)):
            firstSum += c**2.0
            secondSum += np.cos(2.0*np.pi*vector[c])
            n = float(len(vector))
        return [-20.0*np.exp(-0.2*np.sqrt(firstSum/n)) - np.exp(secondSum/n) + 20 + np.e   ]
     
     
     
    def evaluation(self):
        observations=[0]
        return observations
    
    def objectivefunction(self, simulation,evaluation):
        objectivefunction= -spotpy.objectivefunctions.rmse(evaluation = evaluation, simulation = simulation)
        return objectivefunction