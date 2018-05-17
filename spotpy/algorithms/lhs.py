# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python (SPOTPY).
:author: Tobias Houska
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
import numpy as np
import random

class lhs(_algorithm):
    """
    The Latin Hypercube algorithm generates random parameters from their respective
    distribution functions.
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

        super(lhs, self).__init__(*args, **kwargs)

    def sample(self, repetitions):
        """
        test
        """
        print('Starting the LHS algotrithm with '+str(repetitions)+ ' repetitions...')
        self.set_repetiton(repetitions)
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
        matrix = np.empty((repetitions, len(parmin)))
        # Create the LatinHypercube matrix as given in McKay et al. (1979)
        for i in range(int(repetitions)):
            segmentMin = i * segment
            pointInSegment = segmentMin + (random.random() * segment)
            parset = pointInSegment * (parmax - parmin) + parmin
            matrix[i] = parset
        for i in range(len(names)):
            random.shuffle(matrix[:, i])

        # A generator that produces the parameters
        param_generator = ((rep, matrix[rep])
                           for rep in range(int(repetitions)))
        for rep, randompar, simulations in self.repeat(param_generator):
            # A function that calculates the fitness of the run and the manages the database 
            self.postprocessing(rep, randompar, simulations)
        self.final_call()
