'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This example implements the Rosenbrock function into SPOT.  
'''

import numpy as np
import spotpy

        
class spot_setup(object):
    def __init__(self):
        self.params = [spotpy.parameter.Uniform('x',-10,10,1.5,3.0),
                       spotpy.parameter.Uniform('y',-10,10,1.5,3.0),
                       ]
    def parameters(self):
        return spotpy.parameter.generate(self.params)
        
    def simulation(self,vector):      
        x=np.array(vector)
        simulations= [sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)]
        return simulations
        
    def evaluation(self):
        observations=[0]
        print 'type Observations'
        print type(observations)
        return observations
    
    def objectivefunction(self,simulation,evaluation):
        print 'type observation in objectivefunction'
        print type(evaluation)
        objectivefunction=-spotpy.objectivefunctions.rmse(evaluation,simulation)      
        return objectivefunction