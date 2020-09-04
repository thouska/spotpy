'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python (SPOTPY).
:author: Tobias Houska

This example implements the Rosenbrock function into a SPOTPY class.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import rmse
        
class spot_setup(object):
    """
    A 3 dimensional implementation of the Rosenbrock function

    Result at (1,1,1) is 0.
    """
    x = Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='x value of Rosenbrock function')
    y = Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='y value of Rosenbrock function')
    z = Uniform(-10, 10, 1.5, 3.0, -10, 10, doc='z value of Rosenbrock function')
    
    def __init__(self,obj_func=None):
        self.obj_func = obj_func
    
    def simulation(self, vector):
        x=np.array(vector)
        simulations= [sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)]
        return simulations
        
    def evaluation(self):
        observations = [0]
        return observations
    
    def objectivefunction(self, simulation, evaluation):
        #SPOTPY expects to get one or multiple values back, 
        #that define the performence of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = rmse(evaluation,simulation)
        else:
            #Way to ensure on flexible spot setup class
            like = self.obj_func(evaluation,simulation)    
        return like