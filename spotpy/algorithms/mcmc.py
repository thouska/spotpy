# -*- coding: utf-8 -*-
'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
import numpy as np
import time

    
class mcmc(_algorithm):
    """
    This class holds the MarkovChainMonteCarlo (MCMC) algorithm, based on:
    Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H. and Teller, E. (1953) 
    Equation of state calculations by fast computing machines, J. Chem. Phys.
    """

    def __init__(self, *args, **kwargs):
        """
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

        parallel: str
            * seq: Sequentiel sampling (default): Normal iterations on one core of your cpu.
            * mpi: Message Passing Interface: Parallel computing on cluster pcs (recommended for unix os).

        save_sim: boolean
            * True:  Simulation results will be saved
            * False: Simulation results will not be saved
        """
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
        print('Starting the MCMC algotrithm with '+str(repetitions)+ ' repetitions...')
        self.set_repetiton(repetitions)
        # Prepare storing MCMC chain as array of arrays.
        self.nChains = int(nChains)
        #Ensure initialisation of chains and database
        self.burnIn = self.nChains
        # define stepsize of MCMC.        
        self.stepsizes = self.parameter()['step']  # array of stepsizes

        # Metropolis-Hastings iterations.
        self.bestpar=np.array([[np.nan]*len(self.stepsizes)]*self.nChains)
        self.bestlike=[[-np.inf]]*self.nChains
        self.bestsim=[[np.nan]]*self.nChains
        self.accepted=np.zeros(self.nChains)
        self.nChainruns=[[0]]*self.nChains
        self.min_bound, self.max_bound = self.parameter(
        )['minbound'], self.parameter()['maxbound']
        print('Initialize ', self.nChains, ' chain(s)...')
        self.iter=0
        param_generator = ((curChain,self.parameter()['random']) for curChain in range(int(self.nChains)))                
        for curChain,randompar,simulations in self.repeat(param_generator):
            # A function that calculates the fitness of the run and the manages the database 
            like = self.postprocessing(self.iter, randompar, simulations, chains=curChain)
            self.update_mcmc_status(randompar, like, simulations, curChain)
            self.iter+=1

        intervaltime = time.time()
        print('Beginn of Random Walk')
        # Walk through chains
        while self.iter <= repetitions - self.burnIn:
            param_generator = ((curChain,self.get_new_proposal_vector(self.bestpar[curChain])) for curChain in range(int(self.nChains)))                
            for cChain,randompar,simulations in self.repeat(param_generator):
                # A function that calculates the fitness of the run and the manages the database                 
                like = self.postprocessing(self.iter, randompar, simulations, chains=cChain)
                logMetropHastRatio = np.abs(self.bestlike[cChain])/np.abs(like)
                u = np.random.uniform(low=0.3, high=1)
                if logMetropHastRatio>1.0 or logMetropHastRatio>u:
                    self.update_mcmc_status(randompar,like,simulations,cChain)   
                    self.accepted[cChain] += 1  # monitor acceptance
                self.iter+=1                             
                # Progress bar
                acttime = time.time()
            #Refresh MCMC progressbar every two second
            if acttime - intervaltime >= 2 and self.iter >=2:
                text = '%i of %i (best like=%g)' % (
                    self.iter + self.burnIn, repetitions, self.status.objectivefunction)
                text = "Acceptance rates [%] =" +str(np.around((self.accepted)/float(((self.iter-self.burnIn)/self.nChains)),decimals=4)*100).strip('array([])')
                print(text)
                intervaltime = time.time()
        self.final_call()       

