# -*- coding: utf-8 -*-
'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska and Motjaba Sadegh
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
import numpy as np
import random
import time


class dream(_algorithm):
    """
    Implements the DiffeRential Evolution Adaptive Metropolis (DREAM) algorithhm 
    based on:
    Vrugt, J. A. (2016) Markov chain Monte Carlo simulation using the DREAM software package.
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

    def get_regular_startingpoint(self,nChains):
        randompar=self.parameter()['random']        
        for i in range(1000):
            randompar=np.column_stack((randompar,self.parameter()['random']))
        startpoints = []
        for j in range(nChains):
            startpoints.append(np.percentile(randompar,(j+1)/float(nChains+1)*100,axis=1))#,np.amax(randompar,axis=1)
        startpoints = np.array(startpoints)
        for k in range(len(randompar)):
            random.shuffle(startpoints[:, k])
        return startpoints
    
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

    def _get_gamma(self,N):
        #N = Number of parameters
        p = np.random.uniform(low=0,high=1)
        if p >=0.2:
            gamma = 2.38/np.sqrt(2*int(N))#/self.gammalevel
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
                new_parameterset.append(cur_par_set[i] + gamma*np.array(random_par_set1[i]-random_par_set2[i]) + np.random.normal(0,self.eps))
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

    def get_r_hat(self, parameter_array):
        """
        Based on some fancy mathlab code, it return an array [R_stat, MR_stat]
        :param parameter_array: 3 dim array of parameter estimation sets
        :type parameter_array: list
        :return: [R_stat, MR_stat]
        :rtype: list
        """
        n, d, N = parameter_array.shape

        # Use only the last 50% of each chain (vrugt 2009), that means only the half of "d". Cause "d" ist the count
        # of the repetition and we use the d/2 to d of those values which are already not NAN
        whereIsNoNAN = np.logical_not(np.isnan(parameter_array))

        alreadyToNum = np.sum(whereIsNoNAN[0, :, 0])

        if alreadyToNum > 3:
            parameter_array = parameter_array[:, int(np.floor(alreadyToNum / 2)): alreadyToNum, :]
        else:
            # the later functions need some data to work right, so we use in this case 100% of NON NAN values
            parameter_array = parameter_array[:, 0: alreadyToNum, :]

        # I made a big confusion with d, n and  N, I figured it out by tests

        if n > 3:

            mean_chains = np.zeros((n, N))
            for i in range(n):
                for j in range(N):
                    mean_chains[i, j] = np.nanmean(parameter_array[i, :, j])

            B_uni = np.zeros(N)
            for i in range(N):
                B_uni[i] = d * np.nanvar(mean_chains[:, i],

                                      ddof=1)  # make numpy Mathalab like: https://stackoverflow.com/a/27600240/5885054

            var_chains = np.zeros((n, N))
            for i in range(n):
                for j in range(N):
                    var_chains[i, j] = np.nanvar(parameter_array[i, :, j], ddof=1)

            W_uni = np.zeros(N)
            for i in range(N):
                W_uni[i] = np.mean(var_chains[:, i])

            sigma2 = ((d - 1) / d) * W_uni + (1 / d) * B_uni

            whichW_UNIIsNull = W_uni == 0.0
            W_uni[whichW_UNIIsNull] = np.random.uniform(0.1,1,1)

            R_stat = np.sqrt((n + 1) / n * (np.divide(sigma2, W_uni)) - (d - 1) / (n * d))
            
            
#            W_mult = 0
#            for ii in range(n):
#                W_mult = W_mult + np.cov(np.nan_to_num(np.transpose(parameter_array[ii, :, :])), ddof=1)
#
#            W_mult = W_mult / n + 2e-52 * np.eye(N)
#
#            # Note that numpy.cov() considers its input data matrix to have observations in each column,
#            # and variables in each row, so to get numpy.cov() to return what other packages do,
#            # you have to pass the transpose of the data matrix to numpy.cov().
#            # https://stats.stackexchange.com/a/263508/168054
#
#            B_mult = np.cov(np.nan_to_num(np.transpose(mean_chains))) + 2e-52 * np.eye(N)  # 2e-52 avoids problems with eig if var = 0
#            M = np.linalg.lstsq(W_mult, B_mult)
#            R = np.max(np.abs(np.linalg.eigvals(M[0])))
#            MR_stat = np.sqrt((n + 1) / n * R + (d - 1) / d)
            return R_stat#[R_stat, MR_stat]

    def sample(self, repetitions,nChains=5, nCr=3, eps=10e-6, convergence_limit=1.2, runs_after_convergence=100,acceptance_test_option=6):
        print('Starting the DREAM algotrithm with '+str(repetitions)+ ' repetitions...')
        self.set_repetiton(repetitions)
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
        self.gammalevel=1
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
        
        print('Initialize ', self.nChains, ' chain(s)...')
        self.iter=0
        #for i in range(10):
        startpoints = self.get_regular_startingpoint(nChains)
        #param_generator = ((curChain,list(self.parameter()['random'])) for curChain in range(int(self.nChains)))   #TODO: Start with regular interval raster             
        param_generator = ((curChain,list(startpoints[curChain])) for curChain in range(int(self.nChains)))   #TODO: Start with regular interval raster             
        for curChain,par,sim in self.repeat(param_generator):
            like = self.postprocessing(self.iter, par, sim, chains=curChain)
            self.update_mcmc_status(par,like,sim,curChain)
            self.iter+=1
            self.nChainruns[curChain] +=1


        print('Beginn of Random Walk')
        convergence = False
        #Walf through chains
        self.r_hats=[]
        self.eps = eps
        self.CR = []
        for i in range(nCr):
            self.CR.append((i+1)/nCr)
        self.N = len(self.parameter()['random'])
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
                like = self.postprocessing(self.iter, par, sim, chains=cChain)
                #like = self.objectivefunction(
                #    evaluation=self.evaluation, simulation=sim)
                #self.status(self.iter, like, par)
                
#                logMetropHastRatio = np.abs(self.bestlike[cChain])/np.abs(like) #Fast convergence high uncertainty
#                u = np.random.uniform(low=0.0, high=1)
             
#                logMetropHastRatio = like - self.bestlike[cChain] # Slow convergence, low uncertainty
#                u = np.log(np.random.uniform(low=0.0, high=1))

                # set a option which type of comparision should be choose:

                metro_opt=acceptance_test_option

                if metro_opt == 1:
                    logMetropHastRatio = like/self.bestlike[cChain]

                elif metro_opt == 2 or metro_opt == 4:
                    logMetropHastRatio = np.exp(like - self.bestlike[cChain])

                elif metro_opt == 3:
                    # SSR probability evaluation
                    # nrN is defined in this loop so it will increase every step
                    logMetropHastRatio = (like / self.bestlike[cChain]) ** (-nrN * (1 + self._get_gamma(nrN)) / 2)

                elif metro_opt == 5:
                    # SSR probability evaluation, but now weighted with mesurement error
                    # Note that measurement error is single number --> homoscedastic; variance can be taken out of sum sign
                    # SIGMA will be calculated from the orginal data
                    Sigma = np.mean(np.array(self.evaluation)*0.1)
                    logMetropHastRatio = np.exp(-0.5 * (-like + self.bestlike[cChain])/ (Sigma ** 2))  # signs are different because we write -SSR

                elif metro_opt == 6:  # SSR probability evaluation, but now weighted with mesurement error
                    # Note that measurement error is a vector --> heteroscedastic; variance within sum sign  -- see CompDensity.m
                    logMetropHastRatio = np.exp(-0.5 * (-like + self.bestlike[cChain]))  # signs are different because we write -SSR

                u = np.random.uniform(low=0.0, high=1)
             
                if logMetropHastRatio>u:
                    self.update_mcmc_status(par,like,sim,cChain)   
                    self.accepted[cChain] += 1  # monitor acceptance
                    
                    #self.save(like, par, simulations=sim,chains=cChain)
                else:
                    self.update_mcmc_status(self.bestpar[cChain][self.nChainruns[cChain]-1],self.bestlike[cChain],self.bestsim[cChain],cChain)   
                    #self.save(self.bestlike[cChain], self.bestpar[cChain][self.nChainruns[cChain]], 
                    #                     simulations=self.bestsim[cChain],chains=cChain)
                # Progress bar
                
                #acttime = time.time()
                self.iter+=1
                self.nChainruns[cChain] +=1
                #if acttime - intervaltime >= 2 and self.iter >=2 and self.nChainruns[-1] >=3:
                #    self.r_hats.append(self.get_r_hat(self.bestpar))                
                #    #print(self.r_hats[-1])
                #    text = '%i of %i (best like=%g)' % (
                #        self.iter + self.burnIn, repetitions, self.status.objectivefunction)

            r_hat = self.get_r_hat(self.bestpar)
            #self.gammalevel+=.1
            #print((r_hat < 1.2).all())
            self.r_hats.append(r_hat)
            # Refresh progressbar every two seconds
            acttime = time.time()
            if acttime - intervaltime >= 2 and self.iter >=2 and self.nChainruns[-1] >=3:
                #self.r_hats.append(self.get_r_hat(self.bestpar))                
                #text = '%i of %i (best like=%g)' % (
                #    self.iter + self.burnIn, repetitions, self.status.objectivefunction)
                #print(text)
                text = "Acceptance rates [%] =" +str(np.around((self.accepted)/float(((self.iter-self.burnIn)/self.nChains)),decimals=4)*100).strip('array([])')
                print(text)
                text = "Convergence rates =" +str(np.around((r_hat),decimals=4)).strip('array([])')
                print(text)
                intervaltime = time.time()

            if (np.array(r_hat) < convergence_limit).all() and not convergence and self.nChainruns[-1] >=5:
                #Stop sampling
                print('#############')
                print('Convergence has been achieved after '+str(self.iter)+' of '+str(self.repetitions)+' runs! Finally, '+str(runs_after_convergence)+' runs will be additionally sampled to form the posterior distribution')
                print('#############')
                self.repetitions = self.iter + runs_after_convergence
                self.set_repetiton(self.repetitions)
                #self.iter =self.repetitions - runs_after_convergence
                convergence=True
                
        self.final_call()


        #try:
        #    self.datawriter.finalize()
        #except AttributeError:  # Happens if no database was assigned
        #    pass
        #print('End of sampling')
        #text = '%i of %i (best like=%g)' % (
        #    self.status.rep, repetitions, self.status.objectivefunction)
        #print(text)
        #print('Best parameter set')
        #print(self.status.params)
        #text = 'Duration:' + str(round((acttime - starttime), 2)) + ' s'
        #print(text)
        return self.r_hats
