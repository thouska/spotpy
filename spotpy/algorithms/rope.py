# -*- coding: utf-8 -*-
'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska and Alejandro Chamorro-Chavez
'''
from __future__ import unicode_literals, division, absolute_import
from . import _algorithm
import time
import numpy as np
import random


class rope(_algorithm):
    '''
    This class holds the Robust Parameter Estimation (ROPE) algorithm based on
    Bárdossy and Singh (2008).

    Bárdossy, A. and Singh, S. K.:
    Robust estimation of hydrological model parameters,
    Hydrol. Earth Syst. Sci. Discuss., 5(3), 1641–1675, 2008.
    '''

    def __init__(self, spot_setup, dbname=None, dbformat=None,
                 parallel='seq', save_sim=True, save_threshold=-np.inf,sim_timeout = None):
            
        '''
        Input
        ----------
        :param spot_setup: class
            model: function
                Should be callable with a parameter combination of the
                parameter-function and return an list of simulation results (as
                long as evaluation list)
            parameter: function
                When called, it should return a random parameter combination.
                Which can be e.g. uniform or Gaussian
            objectivefunction: function
                Should return the objectivefunction for a given list of a model
                simulation and observation.
            evaluation: function
                Should return the true values as return by the model.
    
        :param dbname: str
            * Name of the database where parameter, objectivefunction value and
            simulation results will be saved.
    
        :param dbformat: str
            * ram: fast suited for short sampling time. no file will be created and
            results are saved in an array.
            * csv: A csv file will be created, which you can import afterwards.
    
        :param parallel: str
            * seq: Sequentiel sampling (default): Normal iterations on one core
            of your cpu.
            * mpc: Multi processing: Iterations on all available cores on your cpu
            (recommended for windows os).
            * mpi: Message Passing Interface: Parallel computing on cluster pcs
            (recommended for unix os).
    
        :param save_sim: boolean
            *True:  Simulation results will be saved
            *False: Simulationt results will not be saved
        '''
        _algorithm.__init__(self, spot_setup, dbname=dbname,
                            dbformat=dbformat, parallel=parallel,
                            save_sim=save_sim, save_threshold=save_threshold,sim_timeout = sim_timeout)

    def create_par(self, min_bound, max_bound):
        return np.random.uniform(low=min_bound, high=max_bound)

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

    def get_best_runs(self, likes, pars, runs, percentage):
        '''
        Returns the best xx% of the runs'''
        return [new_pars for (new_likes, new_pars) in sorted(zip(likes, pars))[int(len(likes) * (1 - percentage)):]]

    def sample(self, repetitions=None, repetitions_first_run=None,
               subsets=5, percentage_first_run=0.10,
               percentage_following_runs=0.10, NDIR=None):
        """
        Samples from the ROPE algorithm.

        Input
        ----------
        repetitions = Number of runs overall, is only used if the user does
        not specify the other arguments, otherwise its overwritten.
        repetitions_first_run = Number of runs in the first rune
        repetitions_following_runs = Number of runs for all following runs
        subsets = number of time the rope algorithm creates a smaller search
        windows for parameters
        percentage_first_run = amount of runs that will be used for the next
        step after the first subset
        percentage_following_runs = amount of runs that will be used for the
        next step after in all following subsets
        NDIR = The number of samples to draw
        """
        # Repetitions_following_runs raus
        # Braucht zu lang (npar >8)
        # wenn mehr parameter produziert werden sollen als reingehen, rechnet er sich tot (ngen>n)
        #Subsets < 5 führt manchmal zu Absturz
        print('Starting the ROPE algotrithm with '+str(repetitions)+ ' repetitions...')
        self.set_repetiton(repetitions)

        if repetitions_first_run is None:
            #Take the first have of the repetitions as burn-in
            first_run = int(repetitions / 2)

        else:
            #Make user defined number of burn-in repetitions
            first_run = repetitions_first_run

        repetitions_following_runs = int((repetitions-first_run) 
                                          / (subsets-1))
        # Needed to avoid an error in integer division somewhere in depth function
        if repetitions_following_runs % 2 != 0:
            repetitions_following_runs+=1

            
        if NDIR is None:
            NDIR = int(repetitions_following_runs / 100)
        self.NDIR = NDIR

        starttime = time.time()
        intervaltime = starttime
        self.min_bound, self.max_bound = self.parameter(
        )['minbound'], self.parameter()['maxbound']
        #randompar = list(self.parameter()['optguess'])
        #simulations = self.model(randompar)
        #like = self.postprocessing(rep, randompar, simulations)

        # Init ROPE with one subset
        likes = []
        pars = []
        
        
        
        
        # Get the names of the parameters to analyse
        names = self.parameter()['name']# distribution        
        parmin, parmax = self.parameter()['minbound'], self.parameter()[
            'maxbound']
        segment = 1 / float(first_run)
        # Get the minimum and maximum value for each parameter from the

        # Create a matrix to store the parameter sets
        matrix = np.empty((first_run, len(parmin)))
        # Create the LatinHypercube matrix as in McKay et al. (1979):
        for i in range(int(first_run)):
            segmentMin = i * segment
            pointInSegment = segmentMin + (random.random() * segment)
            parset = pointInSegment * (parmax - parmin) + parmin
            matrix[i] = parset
        for i in range(len(names)):
            random.shuffle(matrix[:, i])

        # A generator that produces the parameters
        param_generator = ((rep, matrix[rep])
                           for rep in range(int(first_run) - 1))
        for rep, randompar, simulations in self.repeat(param_generator):
            # A function that calculates the fitness of the run and the manages the database 
            like = self.postprocessing(rep, randompar, simulations)
            likes.append(like)
            pars.append(randompar)
            # Progress bar
            acttime = time.time()
            # Refresh progressbar every second
            if acttime - intervaltime >= 2:
                text = '1 Subset: Run %i of %i (best like=%g)' % (
                    rep, first_run, self.status.objectivefunction)
                print(text)
                intervaltime = time.time()



        for subset in range(subsets - 1):
            if subset == 0:
                best_pars = self.get_best_runs(likes, pars, repetitions_following_runs, 
                                               percentage_first_run)
            else:
                best_pars = self.get_best_runs(likes, pars, repetitions_following_runs,
                                               percentage_following_runs)
            valid = False
            trials = 0
            while valid is False and trials < 10 and repetitions_following_runs>1: 
                new_pars = self.programm_depth(best_pars, repetitions_following_runs)
                if len(new_pars) == repetitions_following_runs:
                    valid = True
                else:
                    trials += 1
            pars = []
            likes = []
            print(len(new_pars))
            if(int(repetitions_following_runs) > len(new_pars)):
                repetitions_following_runs = len(new_pars)
            param_generator = (
                (rep, new_pars[rep]) for rep in range(int(repetitions_following_runs)))   
            for rep, ropepar, simulations in self.repeat(param_generator):
                # Calculate the objective function
                like = self.postprocessing(first_run + rep + repetitions_following_runs * subset, ropepar, simulations)
                likes.append(like)
                pars.append(ropepar)

                # Progress bar
                acttime = time.time()
                if repetitions_following_runs is not None:
                    # Refresh progressbar every second
                    if acttime - intervaltime >= 2:
                        text = '%i Subset: Run %i of %i (best like=%g)' % (
                            subset + 2,
                            rep,
                            repetitions_following_runs,
                            self.status.objectivefunction)
                        print(text)
                        intervaltime = time.time()

        self.final_call()
        

    def programm_depth(self, pars, runs):
        X = np.array(pars)

        N, NP = X.shape
        text = str(N) + ' input vectors with ' + str(NP) + ' parameters'
        print(text)

        Ngen = int(runs)  # int(N*(1/self.percentage))
        print(('Generating ' + str(Ngen) + ' parameters:'))

        NPOSI = Ngen   # Number of points to generate

        EPS = 0.00001

        # Find  max and min values
        XMIN = np.zeros(NP)
        XMAX = np.zeros(NP)
        for j in range(NP):
            XMIN[j] = min(X[:, j])
            XMAX[j] = max(X[:, j])

        # Beginn to generate
        ITRY = 1
        IPOS = 1
        LLEN = N

        CL = np.zeros(NP)
        TL = np.zeros(shape=(LLEN, NP))
        #test=[np.zeros(NP)]
        while (IPOS < NPOSI):
            for IM in range(LLEN):   # LLEN=1000 Random Vectors of dim NP
                for j in range(NP):
                    DRAND = np.random.rand()
                    TL[IM, j] = XMIN[j] + DRAND * (XMAX[j] - XMIN[j])
            LNDEP = self.fHDEPTHAB(N, NP, X, TL, EPS, LLEN)
            for L in range(LLEN):
                ITRY = ITRY + 1
                if LNDEP[L] >= 1:
                    #test.append(TL[L, :])
                    CL = np.vstack((CL, TL[L, :]))
                    IPOS = IPOS + 1
            print((IPOS, ITRY))
        #CL=np.array(test)
        #print('##')
        #print(type(CL[0]))
        #print('###')
        #print(type(np.array(test)[0]))
        #print('####')
        #CL=np.array(test)
        CL = np.delete(CL, 0, 0)
        CL = CL[:NPOSI]
        return CL

    def fHDEPTHAB(self, N, NP, X, TL, EPS, LLEN):
        LNDEP = self.fDEP(N, NP, X, TL, EPS, LLEN)
        return LNDEP

    def fDEP(self, N, NP, X, TL, EPS, LLEN):
        LNDEP = np.array([N for i in range(N)])
        for NRAN in range(int(self.NDIR)):

            #       Random sample of size NP
            JSAMP = np.zeros(N)
            I = np.random.randint(1, N)
            if I > N:
                I = N
            JSAMP[0] = I
            NSAMP = 1

            for index in range(NP - 1):
                dirac = 0
                while dirac == 0:
                    dirac = 1
                    L = np.random.randint(1, N + 1)
                    if L > N:
                        L = N
                    for j in range(NSAMP):
                        if L == JSAMP[j]:
                            dirac = 0
                NSAMP = NSAMP + 1
                JSAMP[index + 1] = L

    #       Covariance matrix of the sample
            S = np.zeros(shape=(NP, NP))
            for i in range(NP):
                row = JSAMP[i]

                # too complicated
                # S[i, :] = [X[int(row) - 1,j] for j in range(NP)]
                # S: random sample from X
                S[i, :] = X[int(row) - 1]

            nx, NP = S.shape
            C = np.zeros(shape=(NP, NP))
            y = np.zeros(shape=(2, 2))
            for j in range(NP):
                for k in range(j + 1):
                    y = np.cov(S[:, k], S[:, j])
                    C[j, k] = y[0, 1]
                    C[k, j] = y[0, 1]
            COV = C

            EVALS, EVE = np.linalg.eig(COV)
            arg = np.argsort(EVALS)
            # Eigenvector in the direction of min eigenvalue
            EVECT = EVE[:, arg[0]]

    #       Project all points on the line through theta with direction
            # given by the eigenvector of the smallest eigenvalue, i.e.
            # the direction orthogonal on the hyperplane given by the np-subset
    #       Compute the one-dimensional halfspace depth of theta on this line
            HELP = []
            for L in range(N):
                k = np.dot(EVECT, X[L, :])
                HELP.append(k)
            HELP = np.array(HELP)
            HELP = sorted(HELP)

            ICROSS = 0
            for LU in range(LLEN):
                if (LNDEP[LU] > ICROSS):
                    EKT = 0.
                    NT = 0
                    EKT = EKT + np.dot(TL[LU, :], EVECT)
                    if ((EKT < (HELP[0] - EPS)) or (
                                EKT > (HELP[N - 1] + EPS))):
                        N1 = 0
                    else:
                        N1 = 1
                        N2 = N
                        dirac = 0
                        while dirac == 0:
                            dirac = 1
                            N3 = (N1 + N2) / 2.
                            if (HELP[int(N3)] < EKT):
                                N1 = N3
                            else:
                                N2 = N3
                            if(N2 - N1) > 1:
                                dirac = 0
                    NUMH = N1
                    LNDEP[LU] = min(LNDEP[LU], min(NUMH + NT, N - NUMH))
        return LNDEP
