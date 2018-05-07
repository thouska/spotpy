# -*- coding: utf-8 -*-
'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Patrick Lauer
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
import spotpy
import numpy as np
import time
import random
import itertools


class abc(_algorithm):
    """
    This class holds the Artificial Bee Colony(ABC) algorithm, based on Karaboga (2007).
    D. Karaboga, AN IDEA BASED ON HONEY BEE SWARM FOR NUMERICAL OPTIMIZATION,TECHNICAL REPORT-TR06, Erciyes University, Engineering Faculty, Computer Engineering Department 2005.
    D. Karaboga, B. Basturk, A powerful and Efficient Algorithm for Numerical Function Optimization: Artificial Bee Colony (ABC) Algorithm, Journal of Global Optimization, Volume:39, Issue:3,pp:459-171, November 2007,ISSN:0925-5001 , doi: 10.1007/s10898-007-9149-x

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

        super(abc, self).__init__(*args, **kwargs)


    def sample(self, repetitions, eb=48, a=(1 / 10), peps=0.0001, ownlimit=False, limit=24):
        """


        Parameters
        ----------
        repetitions: int
            maximum number of function evaluations allowed during optimization
        eb: int
            number of employed bees (half of population size)
        a: float
            mutation factor
        peps: float
            Convergence criterium
        ownlimit: boolean
            determines if an userdefined limit is set or not
        limit: int
            sets the limit
        """
        print('Starting the ABC algotrithm with '+str(repetitions)+ ' repetitions...')
        self.set_repetiton(repetitions)
        # Initialize the Progress bar
        #starttime = time.time()
        #intervaltime = starttime
        # Initialize ABC parameters:
        randompar = self.parameter()['random']
        self.nopt = randompar.size
        random.seed()
        if ownlimit == True:
            self.limit = limit
        else:
            self.limit = eb
        lb, ub = self.parameter()['minbound'], self.parameter()['maxbound']
        # Initialization
        work = []
        # Calculate the objective function
        param_generator = (
            (rep, self.parameter()['random']) for rep in range(eb))
        for rep, randompar, simulations in self.repeat(param_generator):
            # Calculate fitness
            like = self.postprocessing(rep, randompar, simulations, chains = 1)
            #like = self.objectivefunction(
            #    evaluation=self.evaluation, simulation=simulations)
            
            # Save everything in the database
            #self.save(like, randompar, simulations=simulations)
            #self.status(rep, like, randompar)
            c = 0
            p = 0
            # (fit_x,x,fit_v,v,limit,normalized fitness)
            work.append([like, randompar, like, randompar, c, p])
            # Progress bar
            #acttime = time.time()
            # get str showing approximate timeleft to end of simulation in H,
            # M, S
            #timestr = time.strftime("%H:%M:%S", time.gmtime(round(((acttime - starttime) /
            #                                                       (rep + 1)) * (repetitions - (rep + 1)))))
            # Refresh progressbar every second
            #if acttime - intervaltime >= 2:
            #    text = '%i of %i (best like=%g) est. time remaining: %s' % (rep, repetitions,
            #                                                                self.status.objectivefunction, timestr)
            #    print(text)
            #    intervaltime = time.time()

        icall = 0
        gnrng = 1e100
        # and criter_change>pcento:
        while icall < repetitions and gnrng > peps:
            psum = 0
        # Employed bee phase
            # Generate new input parameters
            for i, val in enumerate(work):
                k = i
                while k == i:
                    k = random.randint(0, (eb - 1))
                j = random.randint(0, (self.nopt - 1))
                work[i][3][j] = work[i][1][j] + \
                    random.uniform(-a, a) * (work[i][1][j] - work[k][1][j])
                if work[i][3][j] < lb[j]:
                    work[i][3][j] = lb[j]
                if work[i][3][j] > ub[j]:
                    work[i][3][j] = ub[j]
                '''
                #Scout bee phase
                if work[i][4] >= self.limit:
                    work[i][3]=self.parameter()['random']
                    work[i][4]=0
                '''
            # Calculate the objective function
            param_generator = ((rep, work[rep][3]) for rep in range(eb))
            for rep, randompar, simulations in self.repeat(param_generator):
                # Calculate fitness
                clike = self.postprocessing(icall+eb, randompar, simulations, chains = 2)

                #clike = self.objectivefunction(
                #    evaluation=self.evaluation, simulation=simulations)
                if clike > work[rep][0]:
                    work[rep][1] = work[rep][3]
                    work[rep][0] = clike
                    work[rep][4] = 0
                else:
                    work[rep][4] = work[rep][4] + 1
                
                #self.save(
                #    clike, work[rep][3], simulations=simulations, chains=icall)
                #self.status(rep, work[rep][0], work[rep][1])
                icall += 1
            # Probability distribution for roulette wheel selection
            bn = []
            for i, val in enumerate(work):
                psum = psum + (1 / work[i][0])
            for i, val in enumerate(work):
                work[i][5] = ((1 / work[i][0]) / psum)
                bn.append(work[i][5])
            bounds = np.cumsum(bn)
        # Onlooker bee phase
            # Roulette wheel selection
            for i, val in enumerate(work):
                pn = random.uniform(0, 1)
                k = i
                while k == i:
                    k = random.randint(0, eb - 1)
                for t, vol in enumerate(bounds):
                    if bounds[t] - pn >= 0:
                        z = t
                        break
                j = random.randint(0, (self.nopt - 1))
            # Generate new input parameters
                try:
                    work[i][3][j] = work[z][1][j] + \
                        random.uniform(-a, a) * (work[z][1][j] - work[k][1][j])
                except UnboundLocalError:
                    z=0
                    work[i][3][j] = work[z][1][j] + \
                        random.uniform(-a, a) * (work[z][1][j] - work[k][1][j])
                if work[i][3][j] < lb[j]:
                    work[i][3][j] = lb[j]
                if work[i][3][j] > ub[j]:
                    work[i][3][j] = ub[j]
            # Calculate the objective function
            param_generator = ((rep, work[rep][3]) for rep in range(eb))
            for rep, randompar, simulations in self.repeat(param_generator):
                # Calculate fitness
                clike = self.postprocessing(icall+eb, randompar, simulations, chains = 3)
                #clike = self.objectivefunction(
                #    evaluation=self.evaluation, simulation=simulations)
                if clike > work[rep][0]:
                    work[rep][1] = work[rep][3]
                    work[rep][0] = clike
                    work[rep][4] = 0
                else:
                    work[rep][4] = work[rep][4] + 1                
                #self.save(
                #    clike, work[rep][3], simulations=simulations, chains=icall)
                #self.status(rep, work[rep][0], work[rep][1])
                icall += 1
        # Scout bee phase
            for i, val in enumerate(work):
                if work[i][4] >= self.limit:
                    work[i][1] = self.parameter()['random']
                    work[i][4] = 0
                    t, work[i][0], simulations = self.simulate(
                        (icall, work[i][1]))
                    clike = self.postprocessing(icall+eb, randompar, simulations, chains = 4)
                    #clike = self.objectivefunction(
                    #    evaluation=self.evaluation, simulation=simulations)
                    #self.save(
                    #    clike, work[rep][3], simulations=simulations, chains=icall)
                    work[i][0] = clike
                    icall += 1
            gnrng = -self.status.objectivefunction
            #text = '%i of %i (best like=%g) est. time remaining: %s' % (
            #    icall, repetitions, self.status.objectivefunction, timestr)
            #print(text)
            if icall >= repetitions:
                print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
                print('ON THE MAXIMUM NUMBER OF TRIALS ')
                print(repetitions)
                print('HAS BEEN EXCEEDED.')

            if gnrng < peps:
                print(
                    'THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')
        self.final_call()
