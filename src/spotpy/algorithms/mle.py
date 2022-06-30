'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska
'''

from . import _algorithm
import numpy as np



class mle(_algorithm):
    """
    This class holds the Maximum Likelihood (MLE) algorithm, 
    based on a simple uphill method as presented by Houska et al (2015):
    Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L. (2015) 
    SPOTting Model Parameters Using a Ready-Made Python Package, PLoS ONE.
    """

    def __init__(self, *args, **kwargs):
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
            * True:  Simulation results will be saved
            * False: Simulation results will not be saved
         '''
        kwargs['optimization_direction'] = 'maximize'
        kwargs['algorithm_name'] = 'Maximum Likelihood Estimation (MLE) algorithm'
        super(mle, self).__init__(*args, **kwargs)


    def check_par_validity(self, par):
        if len(par) == len(self.min_bound) and len(par) == len(self.max_bound):
            for i in range(len(par)):
                if par[i] < self.min_bound[i]:
                    par[i] = self.min_bound[i]
                if par[i] > self.max_bound[i]:
                    par[i] = self.max_bound[i]
        else:
            print('ERROR Bounds have not the same lenghts as Parameterarray')
        return par

    def sample(self, repetitions):
        self.set_repetiton(repetitions)
        print('Starting the MLE algotrithm with '+str(repetitions)+ ' repetitions...')
        # Define stepsize of MLE
        stepsizes = self.parameter()['step']  # array of stepsizes
        accepted = 0.0
        self.min_bound, self.max_bound = self.parameter(
        )['minbound'], self.parameter()['maxbound']
        # Metropolis-Hastings iterations.
        burnIn = int(repetitions / 10)
        likes = []
        pars = []
        sims = []
        print('burnIn...')
        for i in range(burnIn):
            randompar = self.parameter()['random']
            pars.append(randompar)
            _, _, simulations = self.simulate((i, randompar))
            sims.append(simulations)
            like = self.postprocessing(i, randompar, simulations)
            likes.append(like)


        old_like = max(likes)
        old_par = pars[likes.index(old_like)]
        print('Beginn Random Walk')
        for rep in range(repetitions - burnIn):
            # Suggest new candidate from Gaussian proposal distribution.
            # Use stepsize provided for every dimension.
            new_par = np.random.normal(loc=old_par, scale=stepsizes)
            new_par = self.check_par_validity(new_par)
            _, _, new_simulations = self.simulate((i, new_par))
            new_like = self.postprocessing(rep+burnIn, new_par, new_simulations)
            # Accept new candidate in Monte-Carlo fashing.
            if (new_like > old_like):
                accepted = accepted + 1.0  # monitor acceptance
                old_par = new_par
                old_like = new_like
                #self.status(rep, new_like, new_par)

        self.final_call() 
