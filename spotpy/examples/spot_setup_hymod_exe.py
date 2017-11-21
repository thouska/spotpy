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
import os
from scipy import stats
import sys
        
class spot_setup(object):
    def __init__(self,mean1=-5.0,mean2=5.0,std1=1.0,std2=1.0):

        self.params = [spotpy.parameter.Uniform('cmax',low=1.0 , high=500,  optguess=412.33),
                       spotpy.parameter.Uniform('bexp',low=0.1 , high=2.0,  optguess=0.1725),
                       spotpy.parameter.Uniform('alpha',low=0.1 , high=0.99, optguess=0.8127),
                       spotpy.parameter.Uniform('Ks',low=0.0 , high=0.10, optguess=0.0404),
                       spotpy.parameter.Uniform('Kq',low=0.1 , high=0.99, optguess=0.5592),
                       spotpy.parameter.Uniform('fake1',low=0.1 , high=10, optguess=0.5592),
                       spotpy.parameter.Uniform('fake2',low=0.1 , high=10, optguess=0.5592)

#                       spotpy.parameter.Uniform('likelihood_beta', -.99, .99),
#                       spotpy.parameter.Uniform('likelihood_xsi', 0.11, 10),
#                       spotpy.parameter.Uniform('likelihood_sigma0', 0, 1),
#                       spotpy.parameter.Uniform('likelihood_sigma1', 0, 1),
#                       spotpy.parameter.Uniform('likelihood_phi1', 0, .99),
#                       spotpy.parameter.Uniform('likelihood_muh', 0, 100)
                       ]
        self.curdir = os.getcwd()
        self.owd = os.path.realpath(__file__)+os.sep+'..'
        self.hymod_path = self.owd+os.sep+'hymod_exe'
        self.evals = list(np.genfromtxt(self.hymod_path+os.sep+'bound.txt',skip_header=65)[:,3])[:730]
        self.Factor = 1944 * (1000 * 1000 ) / (1000 * 60 * 60 * 24)
                    
        print(len(self.evals))

    def parameters(self):
        return spotpy.parameter.generate(self.params)
        
    def simulation(self,x):
        os.chdir(self.hymod_path)
        if sys.version_info.major == 2:
            params = file('Param.in', 'w')
        elif sys.version_info.major == 3:
            params = open('Param.in','w')
        else:
            raise Exception("Your python is too old for this example")
        for i in range(len(x)):
            if i == len(x):
                params.write(str(round(x[i],5)))
            else:
                params.write(str(round(x[i],5))+' ')
        params.close()
        os.system('HYMODsilent.exe')
        
        #try: 
        if sys.version_info.major == 2:
            SimRR = file('Q.out', 'r')
        elif sys.version_info.major == 3:
            SimRR = open('Q.out', 'r')
        else:
            raise Exception("Your python is too old for this example")
        simulations=[]
        for i in range(64):
            SimRR.readline()
        for i in range(730):
            val= SimRR.readline()
            #print(i,val)
            simulations.append(float(val)*self.Factor)
        #except:#Assign bad values - model might have crashed
        #    SimRR = 795 * [0]
        os.chdir(self.curdir)
        
        return simulations
        
    def evaluation(self):
        return self.evals
    
    def objectivefunction(self,simulation,evaluation, params=None):
        like = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(evaluation,simulation)     # Works good
        return like