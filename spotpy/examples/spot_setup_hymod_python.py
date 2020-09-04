'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This example implements the python version of hymod into SPOTPY.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from spotpy.parameter import Uniform
from spotpy.objectivefunctions import rmse
from spotpy.examples.hymod_python.hymod import hymod
import os

class spot_setup(object):
    cmax  = Uniform(low=1.0 , high=500,  optguess=412.33)
    bexp  = Uniform(low=0.1 , high=2.0,  optguess=0.1725)
    alpha = Uniform(low=0.1 , high=0.99, optguess=0.8127)
    Ks    = Uniform(low=0.001 , high=0.10, optguess=0.0404)
    Kq    = Uniform(low=0.1 , high=0.99, optguess=0.5592)
    #fake1 =spotpy.parameter.Uniform(low=0.1 , high=10, optguess=0.5592)
    #fake2 =spotpy.parameter.Uniform(low=0.1 , high=10, optguess=0.5592)
        
    def __init__(self, obj_func=None):
        #Just a way to keep this example flexible and applicable to various examples
        self.obj_func = obj_func  
        #Transform [mm/day] into [l s-1], where 1.783 is the catchment area
        self.Factor = 1.783 * 1000 * 1000 / (60 * 60 * 24) 
        #Load Observation data from file
        self.PET,self.Precip   = [], []
        self.date,self.trueObs = [], []
        #Find Path to Hymod on users system
        self.owd = os.path.dirname(os.path.realpath(__file__))
        self.hymod_path = self.owd+os.sep+'hymod_python'
        climatefile = open(self.hymod_path+os.sep+'hymod_input.csv', 'r')
        headerline = climatefile.readline()[:-1]

        #Read model forcing in working storage (this is done only ones)
        if ';' in headerline:
            self.delimiter = ';'
        else:
            self.delimiter = ','
        self.header = headerline.split(self.delimiter)
        for line in climatefile:
            values =  line.strip().split(self.delimiter)
            self.date.append(str(values[0]))
            self.Precip.append(float(values[1]))
            self.PET.append(float(values[2]))
            self.trueObs.append(float(values[3]))
        climatefile.close()

    def simulation(self,x):
        #Here the model is actualy startet with one paramter combination
        data = hymod(self.Precip, self.PET, x[0], x[1], x[2], x[3], x[4])
        sim=[]
        for val in data:
            sim.append(val*self.Factor)
        #The first year of simulation data is ignored (warm-up)
        return sim[366:]
        
    def evaluation(self):
        return self.trueObs[366:]
    
    def objectivefunction(self,simulation,evaluation, params=None):
        #SPOTPY expects to get one or multiple values back, 
        #that define the performence of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = rmse(evaluation,simulation)
        else:
            #Way to ensure flexible spot setup class
            like = self.obj_func(evaluation,simulation)    
        return like