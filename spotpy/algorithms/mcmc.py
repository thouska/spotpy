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

    def check_par_validity(self, par):
        if len(par) == len(self.min_bound) and len(par) == len(self.max_bound):
            for i in range(len(par)):
                if par[i] < self.min_bound[i]:
                    par[i] = self.min_bound[i]
                if par[i] > self.max_bound[i]:
                    par[i] = self.max_bound[i]
        else:
            print('ERROR: Bounds have not the same lenghts as Parameterarray')
        return par

    def check_par_validity_reflect(self, par):
        if len(par) == len(self.min_bound) and len(par) == len(self.max_bound):
            for i in range(len(par)):
                if par[i] < self.min_bound[i]:
                    par[i] = self.min_bound[i] + (self.min_bound[i]- par[i])
                elif par[i] > self.max_bound[i]:
                    par[i] = self.max_bound[i] - (par[i] - self.max_bound[i])

            # Postprocessing if reflecting jumped out of bounds
            for i in range(len(par)):
                if par[i] < self.min_bound[i]:
                    par[i] = self.min_bound[i]
                if par[i] > self.max_bound[i]:
                    par[i] = self.max_bound[i]
        else:
            print('ERROR: Bounds have not the same lenghts as Parameterarray')
        return par
        
    def get_new_proposal_vector(self,old_par):
        new_par = np.random.normal(loc=old_par, scale=self.stepsizes)
        #new_par = self.check_par_validity(new_par)
        new_par = self.check_par_validity_reflect(new_par)
        return new_par

    def update_mcmc_status(self,par,like,sim,cur_chain):  
        self.bestpar[cur_chain]=par
        self.bestlike[cur_chain]=like
        self.bestsim[cur_chain]=sim

            
    def sample(self, repetitions,nChains=1):
        # Prepare storing MCMC chain as array of arrays.
        # define stepsize of MCMC.
        self.repetitions = int(repetitions)
        self.nChains = int(nChains)
        #Ensure initialisation of chains and database
        self.burnIn = self.nChains
        self.stepsizes = self.parameter()['step']  # array of stepsizes
        starttime = time.time()
        intervaltime = starttime
        # Metropolis-Hastings iterations.
        self.bestpar=np.array([[np.nan]*len(self.stepsizes)]*self.nChains)
        self.bestlike=[[-np.inf]]*self.nChains
        self.bestsim=[[np.nan]]*self.nChains
        self.accepted=np.zeros(self.nChains)
        self.nChainruns=[[0]]*self.nChains
        self.min_bound, self.max_bound = self.parameter(
        )['minbound'], self.parameter()['maxbound']
        print('Inititalize ',self.nChains, ' chain(s)...')
        self.iter=0
        param_generator = ((curChain,self.parameter()['random']) for curChain in range(int(self.nChains)))                
        for curChain,par,sim in self.repeat(param_generator):
            
            like = self.objectivefunction(
                evaluation=self.evaluation, simulation=sim)
            
            self.update_mcmc_status(par,like,sim,curChain)
            self.save(like, par, simulations=sim,chains=curChain)
            self.status(self.iter, like, par)
            self.iter+=1
            # Progress bar
            acttime = time.time()
            # Refresh progressbar every second
            if acttime - intervaltime >= 2:
                text = '%i of %i (best like=%g)' % (
                    self.iter, repetitions, self.status.objectivefunction)
                print(text)
                intervaltime = time.time()

        print('Beginn of Random Walk')
        #Walf through chains
        while self.iter <= self.repetitions - self.burnIn:
            param_generator = ((curChain,self.get_new_proposal_vector(self.bestpar[curChain])) for curChain in range(int(self.nChains)))                
            for cChain,par,sim in self.repeat(param_generator):
                
                like = self.objectivefunction(
                    evaluation=self.evaluation, simulation=sim)
                logMetropHastRatio = np.abs(self.bestlike[cChain])/np.abs(like)
                u = np.random.uniform(low=0.3, high=1)
                if logMetropHastRatio>1.0 or logMetropHastRatio>u:
                    self.update_mcmc_status(par,like,sim,cChain)   
                    self.accepted[cChain] += 1  # monitor acceptance
                    self.save(like, par, simulations=sim,chains=cChain)
                else:
                    self.save(self.bestlike[cChain], self.bestpar[cChain], 
                                         simulations=self.bestsim[cChain],chains=cChain)
                self.status(self.iter, like, par)                                
                # Progress bar
                acttime = time.time()
                self.iter+=1
            # Refresh progressbar every second
            if acttime - intervaltime >= 2 and self.iter >=2:
                text = '%i of %i (best like=%g)' % (
                    self.iter + self.burnIn, repetitions, self.status.objectivefunction)
                text = "Acceptance rates [%] =" +str(np.around((self.accepted)/float(((self.iter-self.burnIn)/self.nChains)),decimals=4)*100).strip('array([])')
                print(text)
                intervaltime = time.time()
                


        try:
            self.datawriter.finalize()
        except AttributeError:  # Happens if no database was assigned
            pass
        print('End of sampling')
        text = '%i of %i (best like=%g)' % (
            self.status.rep, repetitions, self.status.objectivefunction)
        print(text)
        print('Best parameter set')
        print(self.status.params)
        text = 'Duration:' + str(round((acttime - starttime), 2)) + ' s'
        print(text)
