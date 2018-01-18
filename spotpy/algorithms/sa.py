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
        print('Starting the SA algotrithm with '+str(repetitions)+ ' repetitions...')
        self.set_repetiton(repetitions)
        # Tini=80#repetitions/100
        # Ntemp=6
        self.min_bound, self.max_bound = self.parameter(
        )['minbound'], self.parameter()['maxbound']
        stepsizes = self.parameter()['step']
        #starttime = time.time()
        #intervaltime = starttime
        Eopt = 999999
        Titer = Tini
        #vmin,vmax = self.find_min_max()
        x = self.parameter()['optguess']
        Xopt = x
        simulations = self.model(x)
        #SimOpt = simulations
        Enew = self.postprocessing(1, x, simulations)
        Eopt = Enew
        #self.save(Eopt, Xopt, simulations=simulations)
        # k=(vmax-vmin)/self.parameter()['step']
        rep = 0
        while (Titer > 0.001 * Tini and rep < repetitions):
            for counter in range(Ntemp):

                if (Enew > Eopt):
                    # print 'Better'
                    Eopt = Enew
                    Xopt = x
                    #SimOpt = simulations
                    Eopt = Enew
                    x = np.random.uniform(low=Xopt - stepsizes, high=Xopt + stepsizes)

                else:
                    accepted = frandom(Enew, Eopt, Titer)
                    if accepted == True:
                        # print Xopt
                        Xopt = x
                        #SimOpt = self.model(x)
                        #Eopt = self.objectivefunction(
                        #    evaluation=self.evaluation, simulation=simulations)
                        x = np.random.uniform(low=Xopt - stepsizes, high=Xopt + stepsizes)

                    else:
                        x = np.random.normal(loc=Xopt, scale=stepsizes)

                x = self.check_par_validity(x)

                simulations = self.model(x)
                Enew = self.postprocessing(rep+1, x, simulations)
                #                self.objectivefunction(
#                    evaluation=self.evaluation, simulation=simulations)
#                
#                self.save(Eopt, Xopt, simulations=SimOpt)
#                self.status(rep, Enew, Xopt)
                rep += 1


            Titer = alpha * Titer
        self.final_call()  
#        text = '%i of %i (best like=%g)' % (
#            rep, repetitions, self.status.objectivefunction)
#        print(text)
#        try:
#            self.datawriter.finalize()
#        except AttributeError:  # Happens if no database was assigned
#            pass
#        text = 'Duration:' + str(round((acttime - starttime), 2)) + ' s'
#        print(text)
#        data = self.datawriter.getdata()
#        return data


def frandom(Enew, Eold, Titer):
    # dE=Enew-Eold
    dE = Eold - Enew
    accepted = False
    if (dE > 0):
        P = np.exp(-(dE) / Titer)  # Boltzmann distr.
        rn = np.random.rand()

        if (rn <= P):   # New configuration accepted
            # print 'accepted'
            accepted = True
    else:
        # print 'else'
        accepted = True
    return accepted


def fgener(param, vmin, vmax, k):         # random displacement
    rv = np.random.rand()
    k = 10
    rd = 2.0 * (rv - 0.5) * param / float(k)
    new = param + rd
    if (new < vmin):
        new = vmin
    if (new > vmax):
        new = vmax
    return new
