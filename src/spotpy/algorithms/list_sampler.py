# -*- coding: utf-8 -*-
'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska
'''
from . import _algorithm
from .. import analyser
class list_sampler(_algorithm):
    """
    This class holds the List sampler, which samples from a given spotpy database
    """
    _excluded_parameter_classes = ()
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
        kwargs['algorithm_name'] = 'List Sampler'
        super(list_sampler, self).__init__(*args, **kwargs)

    def sample(self, repetitions=None):
        """

        Parameters
        ----------
        Optional:
        repetitions: int
            maximum number of function evaluations allowed during sampling
            If not given number if iterations will be determined based on given list
        """
        
        parameters = analyser.load_csv_parameter_results(self.dbname)
        self.dbname=self.dbname+'list'
        if not repetitions:
            repetitions=len(parameters)
        self.set_repetiton(repetitions)
        
        # Initialization
        print('Starting the List sampler with '+str(repetitions)+ ' repetitions...')
        param_generator = ((rep, list(parameters[rep]))
                           for rep in range(int(repetitions)))
        for rep, randompar, simulations in self.repeat(param_generator):
            # A function that calculates the fitness of the run and the manages the database 
            self.postprocessing(rep, list(randompar), simulations)
        self.final_call()
