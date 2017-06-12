# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska
This class holds the MonteCarlo (MC) algorithm.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
import time


class mc(_algorithm):
    '''
    Implements the MonteCarlo algorithm.

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
        super(mc, self).__init__(*args, **kwargs)

    def sample(self, repetitions):
        """
        Samples from the MonteCarlo algorithm.

        Input
        ----------
        repetitions: int 
            Maximum number of runs.  
        """
        # Initialize the Progress bar
        starttime = time.time()
        intervaltime = starttime
        # Select the assumed best parameter set from the user
        #randompar    = list(self.parameter()['optguess'])
        # Run the model with the parameter set
        #simulations  = self.model(randompar)
        # Calculate the objective function
        #like         = self.objectivefunction(simulations,self.evaluation)
        # Save everything in the database
        # self.datawriter.save(like,randompar,simulations=simulations)
        # A generator that produces the parameters
        param_generator = ((rep, self.parameter()['random'])
                           for rep in range(int(repetitions) - 1))
        for rep, randompar, simulations in self.repeat(param_generator):
            # Calculate the objective function
            like = self.objectivefunction(
                evaluation=self.evaluation, simulation=simulations)
            
 
            # Save everything in the database
            self.save(like, randompar, simulations=simulations)
            self.status(rep, like, randompar)
            # Progress bar
            acttime = time.time()

            # get str showing approximate timeleft to end of simulation in H,
            # M, S

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
