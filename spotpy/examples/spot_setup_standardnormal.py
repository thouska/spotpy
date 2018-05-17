'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This example implements the Standard Normal function into SPOT.  
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import spotpy

        
class spot_setup(object):
    def __init__(self,mean=0,std=1):
        self.params = [spotpy.parameter.Uniform('x',-5,5,1.5,3.0)
                       ]
        self.mean=mean
        self.std=std
        
    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self,x):
        simulations= (1.0/(self.std*np.sqrt(2*np.pi)))**((-1.0/2.0)*(((x-self.mean)/self.std)**2))
        return simulations
        
    def evaluation(self):
        observations = [0]
        return observations
    
    def objectivefunction(self, simulation,evaluation):
        objectivefunction = -spotpy.objectivefunctions.rmse(evaluation = evaluation,simulation = simulation)      
        return objectivefunction