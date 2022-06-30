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
from ._algorithm import _algorithm
import numpy as np
import random


class fscabc(_algorithm):
    """
    This class holds the Fitness Scaled Chaotic Artificial Bee Colony (FSCABC) algorithm, 
    based on:
    
    Yudong Zhang, Lenan Wu, and Shuihua Wang (2011). Magnetic Resonance Brain Image 
    Classification by an Improved Artificial Bee Colony Algorithm. 
    Progress In Electromagnetics Research
    
    Yudong Zhang, Lenan Wu, and Shuihua Wang (2013). 
    UCAV Path Planning by Fitness-Scaling Adaptive Chaotic Particle Swarm Optimization. 
    Mathematical Problems in Engineering
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
        kwargs['algorithm_name'] = 'Fitness Scaled Chaotic Artificial Bee Colony (FSCABC) algorithm'
        super(fscabc, self).__init__(*args, **kwargs)

    def mutate(self, r):
        x = 4 * r * (1 - r)
        return x


    def sample(self, repetitions, eb=48, a=(1 / 10), peps=0.0001, kpow=4, limit=None):
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
            convergence criterion    
        kpow: float
            exponent for power scaling method
        limit: int
            sets the limit for scout bee phase
        """
        self.set_repetiton(repetitions)
        print('Starting the FSCABC algotrithm with '+str(repetitions)+ ' repetitions...')
        # Initialize FSCABC parameters:
        parset = self.parameter()
        randompar = parset['random']
        lb, ub = parset['minbound'], parset['maxbound']
        self.nopt = randompar.size
        random.seed()
        lastbackup=0
        if limit == None:
            self.limit = int(eb/2)
        else:
            self.limit = int(limit)
        # Generate chaos
        r = 0.25
        while r == 0.25 or r == 0.5 or r == 0.75:
            r = random.random()
            
        icall = 0
        gnrng = 1e100

        if self.breakpoint == 'read' or self.breakpoint == 'readandwrite':
            data_frombreak = self.read_breakdata(self.dbname)
            icall = data_frombreak[0]
            work = data_frombreak[1]
            gnrng = data_frombreak[2]
            r = data_frombreak[3]
            # Here database needs to be reinvoked
        elif self.breakpoint is None or self.breakpoint == 'write':
            # Initialization
            work = []
            # Calculate the objective function
            param_generator = (
                (rep, self.parameter()['random']) for rep in range(eb))
            for rep, randompar, simulations in self.repeat(param_generator):
                # Calculate fitness
                like = self.postprocessing(rep, randompar, simulations, negativlike=True)
                c = 0
                p = 0
                # (fit_x,x,fit_v,v,limit,normalized fitness)
                work.append([like, randompar, like, randompar, c, p])
                icall +=1
                if self.status.stop:
                    #icall = repetitions
                    print('Stopping samplig')
                    break

        #Bee Phases
        while icall < repetitions and gnrng > peps:
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
                clike = self.postprocessing(icall, randompar, simulations, chains=1, negativlike=True)
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
                
            # Fitness scaling
            bn = []
            csum = 0
            work.sort(key=lambda item: item[0])
            for i, val in enumerate(work):
                work[i][5] = i**kpow
                csum = work[i][5] + csum
            for i, val in enumerate(work):
                work[i][5] = work[i][5] / csum
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
                clike = self.postprocessing(icall, randompar, simulations, chains=2, negativlike=True)
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
                    for g, bound in enumerate(ub):
                        r = self.mutate(r)
                        work[i][1][g] = lb[g] + r * (ub[g] - lb[g])
                    work[i][4] = 0
                    t, work[i][0], simulations = self.simulate(
                        (icall, work[i][1]))
                    clike = self.postprocessing(icall, randompar, simulations, chains=3, negativlike=True)
                    work[i][0] = clike
                    icall += 1
                    if self.status.stop:
                        print('Stopping samplig')
                        break
            gnrng = -self.status.objectivefunction_max

            if self.breakpoint == 'write' or self.breakpoint == 'readandwrite'\
                    and icall >= lastbackup+self.backup_every_rep:
                savework = (icall, work, gnrng, r)
                self.write_breakdata(self.dbname, savework)
                lastbackup = icall
            if icall >= repetitions:
                print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
                print('ON THE MAXIMUM NUMBER OF TRIALS ')
                print(repetitions)
                print('HAS BEEN EXCEEDED.')

            if gnrng < peps:
                print(
                    'THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')
        self.final_call()
