# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds the MaximumLikelihoodEstimation (MLE) algorithm.
'''

from . import _algorithm
import numpy as np
import time


class mle(_algorithm):
    '''
    Implements the Maximum Likelihood Estimation algorithm.
    
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
     '''
    def __init__(self, spot_setup, dbname='test', dbformat='ram', parallel='seq',save_sim=True):

        _algorithm.__init__(self,spot_setup, dbname=dbname, dbformat=dbformat, parallel=parallel,save_sim=save_sim)


    def find_min_max(self):
        randompar=self.parameter()['random']        
        for i in range(1000):
            randompar=np.column_stack((randompar,self.parameter()['random']))
        return np.amin(randompar,axis=1),np.amax(randompar,axis=1)
    
    def check_par_validity(self,par):
        if len(par) == len(self.min_bound) and len(par) == len(self.max_bound):
            for i in range(len(par)):
                if par[i]<self.min_bound[i]: 
                    par[i]=self.min_bound[i]
                if par[i]>self.max_bound[i]:
                    par[i]=self.max_bound[i]
            return par
        else:
            print('ERROR: Bounds have not the same lenghts as Parameterarray')
            return par    
    
    def sample(self, repetitions):       
        # Define stepsize of MCMC.
        stepsizes = self.parameter()['step']  # array of stepsizes
        accepted  = 0.0
        starttime=time.time()
        intervaltime=starttime
        self.min_bound, self.max_bound = self.find_min_max()
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
            sims.append(sim)
            like = self.objectivefunction(sim,self.evaluation)
            likes.append(like)            
            self.datawriter.save(like,par,simulations=sim)
            self.status(i,like,par)
            #Progress bar
            acttime=time.time()
            #Refresh progressbar every second
            if acttime-intervaltime>=2:
                text='%i of %i (best like=%g)' % (i,repetitions,self.status.objectivefunction)
                print(text)
                intervaltime=time.time()
        
        old_like = max(likes)
        old_par =pars[likes.index(old_like)]
        old_simulations=sims[likes.index(old_like)]
        print('Beginn Random Walk')
        for rep in range(repetitions-burnIn):
            # Suggest new candidate from Gaussian proposal distribution.
            new_par = []#np.zeros([len(old_par)])
            for i in range(len(old_par)):
                # Use stepsize provided for every dimension.
                new_par.append(np.random.normal(loc=old_par[i], scale=stepsizes[i]))
            new_par=self.check_par_validity(new_par)
            new_simulations = self.model(new_par)
            new_like = self.objectivefunction(new_simulations,self.evaluation)
            # Accept new candidate in Monte-Carlo fashing.
            if (new_like > old_like):
                self.datawriter.save(new_like,new_par,simulations=new_simulations)
                accepted = accepted + 1.0  # monitor acceptance
                old_par=new_par
                old_simulations=new_simulations
                old_like=new_like
                self.status(rep,new_like,new_par)                      
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
        print('Best parameter set:')
        print(self.status.params)
        text='Duration:'+str(round((acttime-starttime),2))+' s'
        print(text) 
