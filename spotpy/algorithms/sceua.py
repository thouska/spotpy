# -*- coding: utf-8 -*-
'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska and Stijn Van Hoey
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
import numpy as np


class sceua(_algorithm):
    """
    This class holds the Shuffled Complex Evolution Algortithm (SCE-UA) algorithm, 
    based on:
    Duan, Q., Sorooshian, S. and Gupta, V. K. (1994) 
    Optimal use of the SCE-UA global optimization method for calibrating watershed models, J. Hydrol.

    Based on the PYthon package Optimization_SCE
    Copyright (c) 2011 Stijn Van Hoey.
    Restructured and parallelized by Houska et al (2015):
    Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L. (2015) 
    SPOTting Model Parameters Using a Ready-Made Python Package, PLoS ONE.

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
        kwargs['optimization_direction'] = 'minimize'
        kwargs['algorithm_name'] = 'Shuffled Complex Evolution (SCE-UA) algorithm'
        super(sceua, self).__init__(*args, **kwargs)

    def simulate(self, id_params_tuple):
        """This overwrites the simple wrapper function of _algorithms.py
        and makes a two phase mpi parallelization possbile:
        1) burn-in
        2) complex evolution
        """
        
        if not self.repeat.phase:  # burn-in
            return _algorithm.simulate(self, id_params_tuple)

        else:  # complex-evolution
            igs, x, xf, cx, cf, sce_vars = id_params_tuple
            self.npg, self.nopt, self.ngs, self.nspl, self.nps, self.bl, self.bu, self.stochastic_parameters, discarded_runs = sce_vars
            # Partition the population into complexes (sub-populations);
            k1 = np.arange(self.npg, dtype=int)
            k2 = k1 * self.ngs + igs
            cx[k1, :] = x[k2, :]
            cf[k1] = xf[k2]
            # Evolve sub-population igs for self.self.nspl steps:
            likes = []
            sims = []
            pars = []
            for loop in range(self.nspl):
                # Select simplex by sampling the complex according to a linear
                # probability distribution
                lcs = np.array([0] * self.nps)
                lcs[0] = 1
                for k3 in range(1, self.nps):
                    for i in range(1000):
                        lpos = int(np.floor(
                            self.npg + 0.5 - np.sqrt((self.npg + 0.5)**2 - self.npg * (self.npg + 1) * np.random.random())))
                        # check if the element has already been chosen
                        idx = (lcs[0:k3] == lpos).nonzero()
                        if idx[0].size == 0:
                            break
                    lcs[k3] = lpos
                lcs.sort()

                # Construct the simplex:
                s = cx[lcs, :]
                sf = cf[lcs]

                snew, fnew, simulation, discarded_runs = self._cceua(s, sf, discarded_runs)
                likes.append(fnew)
                pars.append(snew)
                sims.append(simulation)
                
                # Replace the worst point in Simplex with the new point:
                s[-1, :] = snew
                sf[-1] = fnew

                # Replace the simplex into the complex;
                cx[lcs, :] = s
                cf[lcs] = sf

                # Sort the complex;
                idx = np.argsort(cf)
                cf = np.sort(cf)
                cx = cx[idx, :]
                
            # Replace the complex back into the population;
            return igs, likes, pars, sims, cx, cf, k1, k2, discarded_runs

    def sample(self, repetitions, ngs=20, kstop=100, pcento=0.0000001, peps=0.0000001, max_loop_inc=None):
        """
        Samples from parameter distributions using SCE-UA (Duan, 2004), 
        converted to python by Van Hoey (2011), restructured and parallelized by Houska et al (2015).

        Parameters
        ----------
        repetitions: int
            maximum number of function evaluations allowed during optimization
        ngs: int
            number of complexes (sub-populations), take more than the number of
            analysed parameters
        kstop: int
            the number of past evolution loops and their respective objective value to assess whether the marginal improvement at the current loop (in percentage) is less than pcento
        pcento: float
            the percentage change allowed in the past kstop loops below which convergence is assumed to be achieved.
        peps: float
            Value of the normalized geometric range of the parameters in the population below which convergence is deemed achieved.
        max_loop_inc: int
            Number of loops executed at max in this function call
        """
        self.set_repetiton(repetitions)
        # Initialize SCE parameters:
        self.ngs = ngs
        randompar = self.parameter()['random']
        self.nopt = randompar.size
        self.npg = 2 * self.nopt + 1
        self.nps = self.nopt + 1
        self.nspl = self.npg
        npt = self.npg * self.ngs
        self.iseed = 1
        self.discarded_runs = 0
        self.bl, self.bu = self.parameter()['minbound'], self.parameter()[
            'maxbound']
        bound = self.bu - self.bl  # np.array
        self.stochastic_parameters = bound != 0
        proceed = True

        # burnin_only, needed to indictat if only the burnin phase should be run
        burnin_only = False

        if self.breakpoint == 'read' or self.breakpoint == 'readandwrite':
            data_frombreak = self.read_breakdata(self.dbname)
            icall = data_frombreak[0]
            x = data_frombreak[1][0]
            xf = data_frombreak[1][1]
            gnrng = data_frombreak[2]

        elif self.breakpoint is None or self.breakpoint == 'write':
            # Create an initial population to fill array x(npt,self.self.nopt):
            x = self._sampleinputmatrix(npt, self.nopt)
            nloop = 0
            icall = 0
            xf = np.zeros(npt)

            print('Starting burn-in sampling...')

            # Burn in
            param_generator = ((rep, x[rep]) for rep in range(int(npt)))
            for rep, randompar, simulations in self.repeat(param_generator):
                # Calculate the objective function
                like = self.postprocessing(icall, randompar, simulations,chains=0)              
                xf[rep] = like
                icall+=1
                if self.status.stop:
                    print('Stopping samplig. Maximum number of repetitions reached already during burn-in')
                    proceed = False
                    break
            # Sort the population in order of increasing function values;
            idx = np.argsort(xf)
            xf = np.sort(xf)
            x = x[idx, :]

            if max_loop_inc == 1:
                burnin_only = True

            print('Burn-in sampling completed...')

        else:
            raise ValueError("Don't know the breakpoint keyword {}".format(self.breakpoint))
        
        # Record the best points;
        bestx = x[0, :]
        bestf = xf[0]

        BESTF = bestf
        BESTX = bestx

        # Computes the normalized geometric range of the parameters
        gnrng = np.exp(
            np.mean(np.log((np.max(x[:, self.stochastic_parameters], axis=0) - np.min(x[:, self.stochastic_parameters], axis=0)) / bound[self.stochastic_parameters])))

        # Check for convergency;
        if self.status.rep >= repetitions:
            print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
            print('ON THE MAXIMUM NUMBER OF TRIALS ')
            print(repetitions)
            print('HAS BEEN EXCEEDED.  SEARCH WAS STOPPED AT TRIAL NUMBER:')
            print(self.status.rep)
            print('OF THE INITIAL LOOP!')

        if gnrng < peps:
            print(
                'THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')




        # Begin evolution loops:
        nloop = 0
        criter = []
        criter_change_pcent = 1e+5
        proceed = True

        # if only burnin, stop the following while loop to be started
        # write brakpoint if only a single generation shall be computed and
        # the main loop will not be executed
        if burnin_only:
            if self.breakpoint == 'write' or self.breakpoint == 'readandwrite':
                work = (self.status.rep, (x, xf), gnrng)
                self.write_breakdata(self.dbname, work)
            proceed = False
            print('ONLY THE BURNIN PHASE WAS COMPUTED')

        else:
            self.repeat.setphase('ComplexEvo')
            print('Starting Complex Evolution...')

        while icall < repetitions and gnrng > peps and criter_change_pcent > pcento and proceed == True:
            nloop += 1
            print ('ComplexEvo loop #%d in progress...' % nloop)
            # Loop on complexes (sub-populations);
            cx = np.zeros((self.npg, self.nopt))
            cf = np.zeros((self.npg))
            remaining_runs = repetitions - self.status.rep
            if remaining_runs <= self.ngs:
                self.ngs = remaining_runs-1
                proceed = False
            
            sce_vars = [self.npg, self.nopt, self.ngs, self.nspl,
                        self.nps, self.bl, self.bu, self.stochastic_parameters, self.discarded_runs]
            param_generator = ((rep, x, xf, cx, cf, sce_vars)
                               for rep in range(int(self.ngs)))
            for igs, likes, pars, sims, cx, cf, k1, k2, discarded_runs in self.repeat(param_generator):
                x[k2, :] = cx[k1, :]
                xf[k2] = cf[k1]
                self.discard_runs = discarded_runs
                for i in range(len(likes)):
                    if not self.status.stop:    
                        like = self.postprocessing(i, pars[i], sims[i], chains=i+1)
                    else:
                        #Collect data from all slaves but do not save
                        proceed=False
                        like = self.postprocessing(i, pars[i], sims[i], chains=i+1, save_run=False)
                        self.discarded_runs+=1
                        print('Skipping saving')
                
            if self.breakpoint == 'write' or self.breakpoint == 'readandwrite'\
              and self.status.rep >= self.backup_every_rep:
                work = (self.status.rep, (x, xf), gnrng)
                self.write_breakdata(self.dbname, work)

            # End of Loop on Complex Evolution;

            # Shuffled the complexes;
            idx = np.argsort(xf)
            xf = np.sort(xf)
            x = x[idx, :]

            # Record the best and worst points;
            bestx = x[0, :]
            bestf = xf[0]

            # appenden en op einde reshapen!!
            BESTX = np.append(BESTX, bestx, axis=0)
            BESTF = np.append(BESTF, bestf)

            # Computes the normalized geometric range of the parameters
            gnrng = np.exp(
                np.mean(np.log((np.max(x[:, self.stochastic_parameters], axis=0) - np.min(x[:, self.stochastic_parameters], axis=0)) / bound[self.stochastic_parameters])))

            criter = np.append(criter, bestf)
            
            # Check for convergency;
            if self.status.rep >= repetitions:
                print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
                print('ON THE MAXIMUM NUMBER OF TRIALS ')
                print(repetitions)
                print('HAS BEEN EXCEEDED.')

            elif gnrng < peps:
                print(
                    'THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')


            elif nloop >= kstop:  # necessary so that the area of high posterior density is visited as much as possible
                print ('Objective function convergence criteria is now being updated and assessed...')
                absolute_change = np.abs(
                    criter[nloop - 1] - criter[nloop - kstop])*100
                denominator = np.mean(np.abs(criter[(nloop - kstop):nloop]))
                if denominator == 0.0:
                    criter_change_pcent = 0.0
                else:
                    criter_change_pcent = absolute_change / denominator
                print ('Updated convergence criteria: %f' % criter_change_pcent)
                if criter_change_pcent <= pcento:
                    print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY LESS THAN THE USER-SPECIFIED THRESHOLD %f' % (
                        kstop, pcento))
                    print(
                        'CONVERGENCY HAS ACHIEVED BASED ON OBJECTIVE FUNCTION CRITERIA!!!')
            elif self.status.stop:
                proceed = False
                break

            # stop, if max number of loop iteration was reached
            elif max_loop_inc and nloop >= max_loop_inc:
                proceed = False
                print('THE MAXIMAL NUMBER OF LOOPS PER EXECUTION WAS REACHED')
                break
            
        # End of the Outer Loops
        print('SEARCH WAS STOPPED AT TRIAL NUMBER: %d' % self.status.rep)
        print('NUMBER OF DISCARDED TRIALS: %d' % self.discarded_runs)
        print('NORMALIZED GEOMETRIC RANGE = %f' % gnrng)
        print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY %f PERCENT' % (
            kstop, criter_change_pcent))

        # reshape BESTX
        #BESTX = BESTX.reshape(BESTX.size // self.nopt, self.nopt)
        self.final_call()
        

    def _cceua(self, s, sf, discarded_runs):
            #  This is the subroutine for generating a new point in a simplex
            #
            #   s(.,.) = the sorted simplex in order of increasing function values
            #   s(.) = function values in increasing order
            #
            # LIST OF LOCAL VARIABLES
            #   sb(.) = the best point of the simplex
            #   sw(.) = the worst point of the simplex
            #   w2(.) = the second worst point of the simplex
            #   fw = function value of the worst point
            #   ce(.) = the centroid of the simplex excluding wo
            #   snew(.) = new point generated from the simplex
            #   iviol = flag indicating if constraints are violated
            #         = 1 , yes
            #         = 0 , no
        constant_parameters = np.invert(self.stochastic_parameters)
        self.nps, self.nopt = s.shape
        alpha = 1.0
        beta = 0.5

        # Assign the best and worst points:
        sw = s[-1, :]
        fw = sf[-1]

        # Compute the centroid of the simplex excluding the worst point:
        ce = np.mean(s[:-1, :], axis=0)

        # Attempt a reflection point
        snew = ce + alpha * (ce - sw)
        snew[constant_parameters] = sw[constant_parameters]
        # Check if is outside the bounds:
        ibound = 0
        s1 = snew - self.bl
        idx = (s1 < 0).nonzero()
        if idx[0].size != 0:
            ibound = 1

        s1 = self.bu - snew
        idx = (s1 < 0).nonzero()
        if idx[0].size != 0:
            ibound = 2

        if ibound >= 1:
            snew = self._sampleinputmatrix(1, self.nopt)[0]

        ##    fnew = functn(self.nopt,snew);
        _, _, simulations = _algorithm.simulate(self, (1, snew))
        like = self.postprocessing(1, snew, simulations, save_run=False, block_print=True)
        discarded_runs+=1
            
        fnew = like

        # Reflection failed; now attempt a contraction point:
        if fnew > fw:
            snew = sw + beta * (ce - sw)
            snew[constant_parameters] = sw[constant_parameters]

            _, _, simulations = _algorithm.simulate(self, (2, snew))
            like = self.postprocessing(2, snew, simulations, save_run=False, block_print=True)
            discarded_runs+=1
            fnew = like

        # Both reflection and contraction have failed, attempt a random point;
            if fnew > fw:
                snew = self._sampleinputmatrix(1, self.nopt)[0]
                _, _, simulations = _algorithm.simulate(self, (3, snew))
                like = self.postprocessing(3, snew, simulations, save_run=False, block_print=True)
                discarded_runs+=1
            fnew = like
        # END OF CCE
        return snew, fnew, simulations, discarded_runs

    def _sampleinputmatrix(self, nrows, npars):
        '''
        Create inputparameter matrix for nrows simualtions,
        for npars with bounds ub and lb (np.array from same size)
        distname gives the initial sampling ditribution (currently one for all parameters)

        returns np.array
        '''
        x = np.zeros((nrows, npars))
        for i in range(nrows):
            x[i, :] = self.parameter()['random']
        return x
