# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds the MarkovChainMonteCarlo (MCMC) algorithm based on Metropolis et al. (1953).

Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H. and Teller, E.: Equation of state calculations by fast computing machines, J. Chem. Phys., 21(6), 1087–1092, 1953.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
import numpy as np
import time

class mcmc(_algorithm):
    '''
    Implements the MarkovChainMonteCarlo algorithm.
    
    Input
    ----------
    spot_setup: class
        model: function 
            Should be callable with a parameter combination of the parameter-function 
            and return an list of simulation results (as long as evaluation list)
        parameter: function
            When called, it should return a random parameter combination. Which can 
            be e.g. uniform or Gaussian
        objectivefunction: function 
            Should return the objectivefunction for a given list of a model simulation and 
            observation.
        evaluation: function
            Should return the true values as return by the model.
            
    dbname: str
        * Name of the database where parameter, objectivefunction value and simulation results will be saved.
    
    dbformat: str
        * ram: fast suited for short sampling time. no file will be created and results are saved in an array.
        * csv: A csv file will be created, which you can import afterwards.        
        
    save_sim: boolean
        *True:  Simulation results will be saved
        *False: Simulationt results will not be saved

    alt_objfun: str or None, default: 'log_p'
        alternative objectivefunction to be used for algorithm
        * None: the objfun defined in spot_setup.objectivefunction is used
        * any str: if str is found in spotpy.objectivefunctions, 
            this objectivefunction is used, else falls back to None 
            e.g.: 'log_p', 'rmse', 'bias', 'kge' etc.
     '''
    def __init__(self, *args, **kwargs):
        if 'alt_objfun' not in kwargs:
            kwargs['alt_objfun'] = 'log_p'
        super(mcmc, self).__init__(*args, **kwargs)
      
    def check_par_validity(self,par):
        if len(par) == len(self.min_bound) and len(par) == len(self.max_bound):
            for i in range(len(par)):
                if par[i]<self.min_bound[i]: 
                    par[i]=self.min_bound[i]
                if par[i]>self.max_bound[i]:
                    par[i]=self.max_bound[i] 
        else:
            print('ERROR: Bounds have not the same lenghts as Parameterarray')
        return par
                    
    def sample(self, repetitions):       
        # Prepare storing MCMC chain as array of arrays.
        # define stepsize of MCMC.
        stepsizes    = self.parameter()['step']  # array of stepsizes
        accepted     = 0.0
        starttime    = time.time()
        intervaltime = starttime
        # Metropolis-Hastings iterations.
        burnIn=int(repetitions/10)
        likes=[]
        pars=[]
        sims=[]
        print('burnIn...')
        for i in range(burnIn):
            par = self.parameter()['random']
            pars.append(par)
            sim = self.model(par)
            like = self.objectivefunction(evaluation = self.evaluation, simulation = sim)
            likes.append(like)
            sims.append(sim)            
            self.datawriter.save(like,par,simulations = sim)
            self.status(i,like,par)
            #Progress bar
            acttime=time.time()
            #Refresh progressbar every second
            if acttime-intervaltime>=2:
                text='%i of %i (best like=%g)' % (i,repetitions,self.status.objectivefunction)
                print(text)
                intervaltime=time.time()
        
        old_like = max(likes)
        index=likes.index(old_like)
        old_par =pars[index]
        old_simulations=sims[index]
        self.min_bound, self.max_bound = self.parameter()['minbound'],self.parameter()['maxbound']
        
        nevertheless=0
        print('Beginn of Random Walk')
        for rep in range(repetitions-burnIn):
            # Suggest new candidate from Gaussian proposal distribution.
            #np.zeros([len(old_par)])
            #Create new paramert combination and check if all parameter are into 
            #the given parameter bounds
            new_par = []
            for i in range(len(old_par)):
                # Use stepsize provided for every dimension.
                new_par.append(np.random.normal(loc=old_par[i], scale=stepsizes[i]))

            new_par=self.check_par_validity(new_par)
            new_simulations = self.model(new_par)
            new_like = self.objectivefunction(evaluation = self.evaluation, simulation = new_simulations)
            self.status(rep,new_like,new_par)      
            # Accept new candidate in Monte-Carlo fashing.
            if (new_like > old_like):
                self.datawriter.save(new_like,new_par,simulations=new_simulations)
                self.status(rep+burnIn,new_like,new_par)                
                accepted = accepted + 1.0  # monitor acceptance
                old_par=new_par
                old_simulations=new_simulations
                old_like=new_like
            else:            
                logMetropHastRatio = new_like - old_like
                u = np.log(np.random.uniform(low=0,high=1)   )
                if u < logMetropHastRatio: #Standard Metropolis decision
                    #if u < 0.85: #Accept nevertheless with 85% probability (Igancio)
                    nevertheless+=1
                    self.datawriter.save(new_like,new_par,simulations=new_simulations)               
                    self.status(rep+burnIn,new_like,new_par)                      
                    accepted = accepted + 1.0  # monitor acceptance
                    old_par=new_par
                    old_simulations=new_simulations
                    old_like=new_like
                else:
                    self.datawriter.save(old_like,old_par,simulations=old_simulations)
            #Progress bar
            acttime=time.time()
            #Refresh progressbar every second
            if acttime-intervaltime>=2:
                text='%i of %i (best like=%g)' % (rep+burnIn,repetitions,self.status.objectivefunction)
                print(text)
                intervaltime=time.time()
        
        try:
            self.datawriter.finalize()
        except AttributeError: #Happens if no database was assigned
            pass
        print('End of sampling')
        text="Acceptance rate = "+str(accepted/repetitions)        
        print(text)
        text='%i of %i (best like=%g)' % (self.status.rep,repetitions,self.status.objectivefunction)
        print(text)
        print('Best parameter set')
        print(self.status.params)
        text='Duration:'+str(round((acttime-starttime),2))+' s'
        print(text)