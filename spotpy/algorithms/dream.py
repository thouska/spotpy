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


class dream(_algorithm):
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
        super(dream, self).__init__(*args, **kwargs)

    def check_par_validity_bound(self, par):
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
                if par[i] > self.max_bound[i]:
                    par[i] = self.max_bound[i] - (par[i] - self.max_bound[i])
        else:
            print('ERROR: Bounds have not the same lenghts as Parameterarray')
        return par

    def _get_gamma(self,N):
        #N = Number of parameters
        p = np.random.uniform(low=0,high=1)
        if p >=0.2:
            gamma = 2.38/np.sqrt(2*int(N))
        else:
            gamma = 1
        return gamma

    def get_other_random_chains(self,cur_chain):
        valid=False        
        while valid == False:         
            random_chain1 = np.random.randint(0,self.nChains)
            random_chain2 = np.random.randint(0,self.nChains)
            if random_chain1!=cur_chain and random_chain2!=cur_chain and random_chain1!=random_chain2:
                valid=True
        return random_chain1, random_chain2
        
    def get_new_proposal_vector(self,cur_chain,newN,nrN):
        gamma = self._get_gamma(nrN)
        random_chain1,random_chain2 = self.get_other_random_chains(cur_chain)
        new_parameterset=[]        
        #position = self.chain_samples-1#self.nChains*self.chain_samples+self.chain_samples+cur_chain-1
        cur_par_set = self.bestpar[cur_chain][self.nChainruns[cur_chain]-1]
        random_par_set1 = self.bestpar[random_chain1][self.nChainruns[random_chain1]-1]
        random_par_set2 = self.bestpar[random_chain2][self.nChainruns[random_chain2]-1]
                
        for i in range(self.N):#Go through parameters
            
            if newN[i] == True:
                new_parameterset.append(cur_par_set[i] + gamma*np.array(random_par_set1[i]-random_par_set2[i]) + np.random.uniform(0,self.eps))
            else:
                new_parameterset.append(cur_par_set[i])
                
        new_parameter=self.check_par_validity_reflect(new_parameterset)        
        #new_parameter=self.check_par_validity_bound(new_parameterset)        
        return new_parameter
        
#        new_par = np.random.normal(loc=old_par, scale=self.stepsizes)
#        new_par = self.check_par_validity_reflect(new_par)
#        return new_par

    def update_mcmc_status(self,par,like,sim,cur_chain):  
        self.bestpar[cur_chain][self.nChainruns[cur_chain]]=par
        self.bestlike[cur_chain]=like
        self.bestsim[cur_chain]=sim
        
    def get_r_hat(self,parameter_array): # TODO: Use only the last 50% of each chain (vrugt 2009)
        # Calculates the \hat{R}-convergence diagnostic
        # ----------------------------------------------------
        # For more information please refer to: 
        # Gelman, A. and D.R. Rubin, (1992) Inference from Iterative Simulation 
        #      Using Multiple chain, Statistical Science, Volume 7, Issue 4, 
        #      457-472.
        # Brooks, S.P. and A. Gelman, (1998) General Methods for Monitoring 
        #      Convergence of Iterative Simulations, Journal of Computational and 
        #      Graphical Statistics. Volume 7, 434-455. Note that this function 
        #      returns square-root definiton of R (see Gelman et al., (2003), 
        #      Bayesian Data Analsyis, p. 297).
        # ----------- DREAM Manual -----            
        # Written by Jasper A. Vrugt
        # Los Alamos, August 2007
        # Translated into Python by Tobias Houska in March 2016
        

        n = self.nChainruns[0] #->hope that the first chain is representative...
        m = self.nChains
        N = self.nr_of_pars#number of parameters #TODO: Adjust for more than 1 parameter
        #x = cur_parameter
        r_hats =[]
        try:
            for x in range(self.nr_of_pars):
                N = m #chains
                T = n #chain samples
                T2 = int(T/2.0) # Analyses just the second half of the chains
                sums2=[]
                cmeans=[]
                for i in range(N):
                    c_mean = (2.0/(T-2.0))*np.sum(parameter_array[i][T2:self.nChainruns[i]-1][x])
                    cmeans.append(c_mean)                
                    sums1=[]                
                    for j in range(T2):
                        sums1.append((parameter_array[i][T2+j][x]-c_mean)**2.0)
                    sums2.append(np.sum(sums1))
                W  = 2.0/(N*(T-2.0))*np.sum(sums2)
                sums =[]
                v_mean = 1.0/N * np.sum(cmeans)
                for i in range(N):
                    sums.append((cmeans[i]-v_mean)**2.0)
                    
                B  = (1.0 / 2.0*(N-1.0))*np.sum(sums)*T
                s2 = ((T-2.0)/T) * W + 2.0/T*B
                R = np.sqrt(((N+1.0)/N)*s2/W-(T-2.0)/(N*T))
                r_hats.append(R)
            return r_hats
        except (ZeroDivisionError, IndexError) as e:
            return [np.inf]*self.nr_of_pars
           
    def sample(self, repetitions,nChains=5, nCr=3, eps=10e-6):
        if nChains <3:
            print('Please use at least n=3 chains!')
            return None
        # Prepare storing MCMC chain as array of arrays.
        # define stepsize of MCMC.
        self.repetitions = int(repetitions)
        self.nChains = int(nChains)
        #Ensure initialisation of chains and database
        self.burnIn = self.nChains
        self.stepsizes = self.parameter()['step']  # array of stepsizes
        self.nr_of_pars = len(self.stepsizes)
        starttime = time.time()
        intervaltime = starttime
        # Metropolis-Hastings iterations.
        self.bestpar=np.array([[[np.nan]*self.nr_of_pars]*self.repetitions]*self.nChains)
        #[0]->chain    #[0][0]->parameter     #[0][0][0]->repetitons
        self.bestlike=[[-np.inf]]*self.nChains
        self.bestsim=[[np.nan]]*self.nChains
        self.accepted=np.zeros(self.nChains)
        self.nChainruns=[0]*self.nChains
        self.min_bound, self.max_bound = self.parameter(
        )['minbound'], self.parameter()['maxbound']
        
        #firstcall = True
        
        print('Inititalize ',self.nChains, ' chain(s)...')
        self.iter=0
        param_generator = ((curChain,list(self.parameter()['random'])) for curChain in xrange(int(self.nChains)))                
        for curChain,par,sim in self.repeat(param_generator):
            
            like = self.objectivefunction(
                evaluation=self.evaluation, simulation=sim)

#            if firstcall==True:
#                self.initialize_database(par, self.parameter()['name'], sim, like)
#                firstcall=False
            self.update_mcmc_status(par,like,sim,curChain)
            self.save(like, par, simulations=sim,chains=curChain)
            self.status(self.iter, like, par)
            self.iter+=1
            self.nChainruns[curChain] +=1
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
        self.r_hats=[]
        self.eps           = eps 
        self.CR = []
        for i in range(nCr):
            self.CR.append((i+1)/nCr)
        self.N             = len(self.parameter()['random'])
        nrN=1
        newN = [True]*self.N
        while self.iter <= self.repetitions - self.burnIn:
            param_generator = ((curChain,self.get_new_proposal_vector(curChain,newN,nrN)) for curChain in range(int(self.nChains)))                
            for cChain,par,sim in self.repeat(param_generator):
                pCr = np.random.randint(0,nCr)
                ids=[]         
                for i in range(self.N):
                    ids.append(np.random.uniform(low=0,high=1))
                newN = []
                nrN  = 0
                for i in range(len(ids)):
                    if ids[i] < self.CR[pCr]:
                        newN.append(True)
                        nrN+=1
                    else:
                        newN.append(False)
                if nrN == 0:
                    ids=[np.random.randint(0,self.N)]
                    nrN=1
                #print(self.bestpar[cChain][self.nChainruns[cChain]-1])
                like = self.objectivefunction(
                    evaluation=self.evaluation, simulation=sim)
                self.status(self.iter, like, par)
                logMetropHastRatio = np.abs(self.bestlike[cChain])/np.abs(like)
                u = np.random.uniform(low=0.5, high=1)
             
                if logMetropHastRatio>u:
                    self.update_mcmc_status(par,like,sim,cChain)   
                    self.accepted[cChain] += 1  # monitor acceptance
                    self.save(like, par, simulations=sim,chains=cChain)
                else:
                    self.update_mcmc_status(self.bestpar[cChain][self.nChainruns[cChain]-1],self.bestlike[cChain],self.bestsim[cChain],cChain)   
                    self.save(self.bestlike[cChain], self.bestpar[cChain][self.nChainruns[cChain]], 
                                         simulations=self.bestsim[cChain],chains=cChain)
                # Progress bar
                
                acttime = time.time()
                self.iter+=1
                self.nChainruns[cChain] +=1
                if acttime - intervaltime >= 2 and self.iter >=2 and self.nChainruns[-1] >=3:
                    self.r_hats.append(self.get_r_hat(self.bestpar))                
                    print(self.r_hats[-1])
                    text = '%i of %i (best like=%g)' % (
                        self.iter + self.burnIn, repetitions, self.status.objectivefunction)

            self.r_hats.append(self.get_r_hat(self.bestpar))
            # Refresh progressbar every second
            if acttime - intervaltime >= 2 and self.iter >=2 and self.nChainruns[-1] >=3:
                self.r_hats.append(self.get_r_hat(self.bestpar))                
                print(self.r_hats[-1])
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
        return self.r_hats
