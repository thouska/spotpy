# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds the LatinHyperCube algorithm based on McKay et al. (1979):

McKay, M. D., Beckman, R. J. and Conover, W. J.: Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code, Technometrics, 21(2), 239–245, doi:10.1080/00401706.1979.10489755, 1979.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
from .. import database
import numpy as np
import random
import time


class lhs(_algorithm):
    '''
    Implements the LatinHyperCube algorithm.

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
        *True:  Simulation results will be saved
        *False: Simulationt results will not be saved
     '''

    def __init__(self, *args, **kwargs):
        kwargs['dbinit'] = False
        super(lhs, self).__init__(*args, **kwargs)

    def sample(self, repetitions):
        """
        Samples from the LatinHypercube algorithm.

        Input
        ----------
        repetitions: int 
            Maximum number of runs.
        """
        print('Creating LatinHyperCube Matrix')
        # Get the names of the parameters to analyse
        names = self.parameter()['name']
        # Define the jump size between the parameter
        segment = 1 / float(repetitions)
        # Get the minimum and maximum value for each parameter from the
        # distribution
        parmin, parmax = self.parameter()['minbound'], self.parameter()[
            'maxbound']

        # Create an matrx to store the parameter sets
        matrx = np.empty((repetitions, len(parmin)))
        # Create the LatinHypercube matrx as in McKay et al. (1979):
        for i in range(int(repetitions)):
            segmentMin = i * segment
            pointInSegment = segmentMin + (random.random() * segment)
            parset = pointInSegment * (parmax - parmin) + parmin
            matrx[i] = parset
        for i in range(len(names)):
            random.shuffle(matrx[:, i])

        print('Start sampling')
        starttime = time.time()
        intervaltime = starttime
        # A generator that produces the parameters
        # param_generator = iter(matrx)
        firstcall = True
        param_generator = ((rep, matrx[rep])
                           for rep in range(int(repetitions) - 1))
        for rep, randompar, simulations in self.repeat(param_generator):
            # Calculate the objective function
            like = self.objectivefunction(
                evaluation=self.evaluation, simulation=simulations)
            if firstcall==True:
                self.initialize_database(randompar, self.parameter()['name'], simulations, like)
                firstcall=False            
            self.status(rep, like, randompar)
            # Save everything in the database
            self.datawriter.save(like, randompar, simulations=simulations)
            # Progress bar
            acttime = time.time()
            
            #create string of the aproximate time left to complete all runs
            timestr = time.strftime("%H:%M:%S", time.gmtime(round(((acttime - starttime) /
                       (rep + 1)) * (repetitions - (rep + 1)))))
            # Refresh progressbar every second 
            if acttime - intervaltime >= 2:
                text = '%i of %i (best like=%g) est. time remaining: %s' % (rep, repetitions,
                    self.status.objectivefunction, timestr)
                print(text)
                intervaltime = time.time()
        self.repeat.terminate()

        try:
            self.datawriter.finalize()
        except AttributeError:  # Happens if no database was assigned
            pass
        print('End of sampling')
        text = '%i of %i (best like=%g)' % (
            self.status.rep, repetitions, self.status.objectivefunction)
        print(text)
        print('Best parameter set')
        print(self.status.params)
        text = 'Duration:' + str(round((acttime - starttime), 2)) + ' s'
        print(text)
