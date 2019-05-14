# -*- coding: utf-8 -*-
'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska and Alejandro Chamorro-Chavez
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
import numpy as np


class sa(_algorithm):
    """
    This class holds the Simulated Annealing (SA) algorithm based on:
    
    Kirkpatrick, S., Gelatt, C. D., Vecchi, M. P. and others (2013). 
    Optimization by simmulated annealing, science.
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
        kwargs['optimization_direction'] = 'maximize'
        kwargs['algorithm_name'] = 'Simulated Annealing (SA) algorithm'
        super(sa, self).__init__(*args, **kwargs)
        
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

    def sample(self, repetitions, Tini=80, Ntemp=50, alpha=0.99):
        """
        Samples from the MonteCarlo algorithm.

        Input
        ----------
        repetitions: int 
            Maximum number of runs.  
        """
        self.set_repetiton(repetitions)
        print('Starting the SA algotrithm with '+str(repetitions)+ ' repetitions...')
        self.min_bound, self.max_bound = self.parameter(
        )['minbound'], self.parameter()['maxbound']
        stepsizes = self.parameter()['step']
        Eopt = 999999
        Titer = Tini
        x = self.parameter()['optguess']
        Xopt = x
        _, _, simulations = self.simulate((1, x))
        Enew = self.postprocessing(1, x, simulations)
        Eopt = Enew
        rep = 1 # Because the model has been started once already
        while (Titer > 0.001 * Tini and rep < repetitions):
            for counter in range(Ntemp):

                if (Enew > Eopt): # Run was better
                    Eopt = Enew
                    Xopt = x
                    Eopt = Enew
                    x = np.random.uniform(low=Xopt - stepsizes, high=Xopt + stepsizes)

                else:
                    accepted = frandom(Enew, Eopt, Titer)
                    if accepted == True:
                        Xopt = x
                        x = np.random.uniform(low=Xopt - stepsizes, high=Xopt + stepsizes)

                    else:
                        x = np.random.normal(loc=Xopt, scale=stepsizes)

                x = self.check_par_validity(x)

                _, _, simulations = self.simulate((rep+1, x))
                Enew = self.postprocessing(rep+1, x, simulations)
                rep += 1
                if self.status.stop:
                    break


            Titer = alpha * Titer
        self.final_call()  


def frandom(Enew, Eold, Titer):
    dE = Eold - Enew
    accepted = False
    if (dE > 0):
        P = np.exp(-(dE) / Titer)  # Boltzmann distr.
        rn = np.random.rand()

        if (rn <= P):   # New configuration accepted
            accepted = True
    else:
        accepted = True
    return accepted