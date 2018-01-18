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
import time
import random



class fscabc(_algorithm):
    """
    This class holds the Fitness Scaled Chaotic Artificial Bee Colony(FSCABC) algorithm, 
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

        super(fscabc, self).__init__(*args, **kwargs)

    def mutate(self, r):
        x = 4 * r * (1 - r)
        return x


    def sample(self, repetitions, eb=48, a=(1 / 10), peps=0.0001, kpow=5, ownlimit=False, limit=24):
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
        ownlimit: boolean
            determines if an userdefined limit is set or not
        limit: int
            sets the limit for scout bee phase
        breakpoint: None, 'write', 'read' or 'readandwrite'
            None does nothing, 'write' writes a breakpoint for restart as specified in backup_every_rep, 'read' reads a breakpoint file with dbname + '.break', 'readandwrite' does both
        backup_every_rep: int
            writes a breakpoint after every generation, if more at least the specified number of samples are carried out after writing the last breakpoint
        """
        print('Starting the FSCABC algotrithm with '+str(repetitions)+ ' repetitions...')
        self.set_repetiton(repetitions)
        # Initialize the progress bar
        starttime = time.time()
        intervaltime = starttime
        # Initialize FSCABC parameters:
        randompar = self.parameter()['random']
        self.nopt = randompar.size
        random.seed()
        lastbackup=0
        if ownlimit == True:
            self.limit = limit
        else:
            self.limit = eb
        lb, ub = self.parameter()['minbound'], self.parameter()['maxbound']
        # Generate chaos
        r = 0.25
        while r == 0.25 or r == 0.5 or r == 0.75:
            r = random.random()
            
        icall = 0
        gnrng = 1e100
        # and criter_change>pcento:

        if self.breakpoint == 'read' or self.breakpoint == 'readandwrite':
            data_frombreak = self.read_breakdata(self.dbname)
            icall = data_frombreak[0]
            work = data_frombreak[1]
            gnrng = data_frombreak[2]
            r = data_frombreak[3]
            acttime = time.time()
            # Here database needs to be reinvoked
        elif self.breakpoint is None or self.breakpoint == 'write':
            # Initialization
            work = []
            # Calculate the objective function
            param_generator = (
                (rep, self.parameter()['random']) for rep in range(eb))
            for rep, randompar, simulations in self.repeat(param_generator):
                # Calculate fitness
                like = self.postprocessing(rep, randompar, simulations)
                #like = self.objectivefunction(
                #    evaluation=self.evaluation, simulation=simulations)

                # Save everything in the database
                #self.save(like, randompar, simulations=simulations)
                
                # Update status information (always do that after saving)
                #self.status(rep, like, randompar)

                c = 0
                p = 0
                # (fit_x,x,fit_v,v,limit,normalized fitness)
                work.append([like, randompar, like, randompar, c, p])
                # Progress bar
                #acttime = time.time()
                # get str showing approximate timeleft to end of simulation in H,
                # M, S
#                timestr = time.strftime("%H:%M:%S", time.gmtime(round(((acttime - starttime) /
#                                                                    (rep + 1)) * (repetitions - (rep + 1)))))
                # Refresh progressbar every second
#                if acttime - intervaltime >= 2:
#                    text = '%i of %i (best like=%g) est. time remaining: %s' % (rep, repetitions,
#                                                                                self.status.objectivefunction, timestr)
#                    print(text)
#                    intervaltime = time.time()


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
                clike = self.postprocessing(icall, randompar, simulations, chains=1)
                #clike = self.objectivefunction(
                #    evaluation=self.evaluation, simulation=simulations)
                if clike > work[rep][0]:
                    work[rep][1] = work[rep][3]
                    work[rep][0] = clike
                    work[rep][4] = 0
                else:
                    work[rep][4] = work[rep][4] + 1
                
                # Save everything in the database
                #self.save(
                #    clike, work[rep][3], simulations=simulations, chains=icall)
                
                # Update status information (always do that after saving)
                #self.status(rep, work[rep][0], work[rep][1])
                icall += 1
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
                clike = self.postprocessing(icall, randompar, simulations, chains=2)
                #clike = self.objectivefunction(
                #    evaluation=self.evaluation, simulation=simulations)
                if clike > work[rep][0]:
                    work[rep][1] = work[rep][3]
                    work[rep][0] = clike
                    work[rep][4] = 0
                else:
                    work[rep][4] = work[rep][4] + 1
                #self.status(rep, work[rep][0], work[rep][1])
                #self.save(
                #    clike, work[rep][3], simulations=simulations, chains=icall)
                icall += 1
        # Scout bee phase
            for i, val in enumerate(work):
                if work[i][4] >= self.limit:
                    for g, bound in enumerate(ub):
                        r = self.mutate(r)
                        work[i][1][g] = lb[g] + r * (ub[g] - lb[g])
                    work[i][4] = 0
                    t, work[i][0], simulations = self.simulate(
                        (icall, work[i][1]))
                    clike = self.postprocessing(icall, randompar, simulations, chains=3)
                    #clike = self.objectivefunction(
                    #    evaluation=self.evaluation, simulation=simulations)
                    #self.save(
                    #    clike, work[rep][3], simulations=simulations, chains=icall)
                    work[i][0] = clike
                    icall += 1
            gnrng = -self.status.objectivefunction
            #text = '%i of %i (best like=%g)' % (
            #    icall, repetitions, self.status.objectivefunction)
            #print(text)
            if self.breakpoint == 'write' or self.breakpoint == 'readandwrite'\
                    and icall >= lastbackup+self.backup_every_rep:
                work = (icall, work, gnrng, r)
                self.write_breakdata(self.dbname, work)
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

