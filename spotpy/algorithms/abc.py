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
import numpy as np
import random


class abc(_algorithm):
    """
    This class holds the Artificial Bee Colony (ABC) algorithm, based on Karaboga (2007).
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
        kwargs['optimization_direction'] = 'maximize'
        kwargs['algorithm_name'] = 'Artificial Bee Colony (ABC) algorithm'
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
        self.set_repetiton(repetitions)
        print('Starting the ABC algotrithm with '+str(repetitions)+ ' repetitions...')
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
        icall = 0
        gnrng = 1e100
        # Calculate the objective function
        param_generator = (
            (rep, self.parameter()['random']) for rep in range(eb))
        for rep, randompar, simulations in self.repeat(param_generator):
            # Calculate fitness
            like = self.postprocessing(rep, randompar, simulations, chains = 1, negativlike=True)
            c = 0
            p = 0
            work.append([like, randompar, like, randompar, c, p])
            icall +=1
            if self.status.stop:
                print('Stopping sampling')
                break

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

            # Calculate the objective function
            param_generator = ((rep, work[rep][3]) for rep in range(eb))
            for rep, randompar, simulations in self.repeat(param_generator):
                # Calculate fitness
                clike = self.postprocessing(icall+eb, randompar, simulations, chains = 2, negativlike=True)
                if clike > work[rep][0]:
                    work[rep][1] = work[rep][3]
                    work[rep][0] = clike
                    work[rep][4] = 0
                else:
                    work[rep][4] = work[rep][4] + 1                
                icall += 1
                if self.status.stop:
                    print('Stopping samplig')
                    break            # Probability distribution for roulette wheel selection
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
                clike = self.postprocessing(icall+eb, randompar, simulations, chains = 3, negativlike=True)
                if clike > work[rep][0]:
                    work[rep][1] = work[rep][3]
                    work[rep][0] = clike
                    work[rep][4] = 0
                else:
                    work[rep][4] = work[rep][4] + 1                
                icall += 1
                if self.status.stop:
                    print('Stopping samplig')
                    break
        # Scout bee phase
            for i, val in enumerate(work):
                if work[i][4] >= self.limit:
                    work[i][1] = self.parameter()['random']
                    work[i][4] = 0
                    t, work[i][0], simulations = self.simulate(
                        (icall, work[i][1]))
                    clike = self.postprocessing(icall+eb, randompar, simulations, chains = 4, negativlike=True)
                    work[i][0] = clike
                    icall += 1
                    if self.status.stop:
                        print('Stopping samplig')
                        break
            gnrng = -self.status.objectivefunction_max
            if icall >= repetitions:
                print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
                print('ON THE MAXIMUM NUMBER OF TRIALS ')
                print(repetitions)
                print('HAS BEEN EXCEEDED.')

            if gnrng < peps:
                print(
                    'THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE AT RUN')
                print(icall)
        self.final_call()
