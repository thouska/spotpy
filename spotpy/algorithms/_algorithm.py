# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds the standards for every algorithm.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from spotpy import database, objectivefunctions
import numpy as np
import time
import os

class _RunStatistic(object):
    """
    this class checks for each run if the objectivefunction got better and holds the 
    best parameter set.
    Every _algorithm has an object of this class as status.
    Usage:
    status = _RunStatistic()
    status(rep,like,params)

    """

    def __init__(self):
        self.rep = 0
        self.params = None
        self.objectivefunction = -1e308
        self.bestrep = 0
        self.starttime = time.time()
        self.last_print = time.time()
        
        self.repetitions = None


    def __call__(self, rep, objectivefunction, params):
        self.curparmeterset = params
        self.rep+=1
        if type(objectivefunction) == type([]):
            if objectivefunction[0] > self.objectivefunction:
                # Show only the first best objectivefunction when working with
                # more than one objectivefunction
                self.objectivefunction = objectivefunction[0]
                self.params = params
                self.bestrep = self.rep
        else:
            if objectivefunction > self.objectivefunction:
                self.params = params
                self.objectivefunction = objectivefunction
                self.bestrep = self.rep
        self.print_status()
            #return True
        #return False

    def print_status(self):
        # get str showing approximate timeleft to end of simulation in H, M, S
        acttime = time.time()
        # Refresh progressbar every two second
        if acttime - self.last_print >= 2:
            avg_time_per_run = (acttime - self.starttime) / (self.rep + 1)
            timestr = time.strftime("%H:%M:%S", time.gmtime(round(avg_time_per_run * (self.repetitions - (self.rep + 1)))))
                    
            text = '%i of %i (best like=%g) est. time remaining: %s' % (self.rep, self.repetitions,
                                                                        self.objectivefunction, timestr)
            print(text)
            self.last_print = time.time()
        
    def __repr__(self):
        return 'Best objectivefunction: %g' % self.objectivefunction


class _algorithm(object):
    """
    Implements an algorithm.

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
        Name of the database where parameter, objectivefunction value and simulation 
        results will be saved.
    dbformat: str
         ram: fast suited for short sampling time. no file will be created and results are saved in an array.
        csv: A csv file will be created, which you can import afterwards.        
    parallel: str
        seq: Sequentiel sampling (default): Normal iterations on one core of your cpu.
        mpc: Multi processing: Iterations on all available cores on your (single) pc
        mpi: Message Passing Interface: Parallel computing on high performance computing clusters, py4mpi needs to be installed

    alt_objfun: str or None, default: 'rmse'
        alternative objectivefunction to be used for algorithm
        * None: the objfun defined in spot_setup.objectivefunction is used
        * any str: if str is found in spotpy.objectivefunctions, 
            this objectivefunction is used, else falls back to None 
            e.g.: 'log_p', 'rmse', 'bias', 'kge' etc.

    """

    def __init__(self, spot_setup, dbname=None, dbformat=None, dbinit=True,
                 parallel='seq', save_sim=True, alt_objfun=None, breakpoint=None, backup_every_rep=100):
        # Initialize the user defined setup class
        self.setup = spot_setup
        self.model = self.setup.simulation
        self.parameter = self.setup.parameters
        self.parnames = self.parameter()['name']
        # use alt_objfun if alt_objfun is defined in objectivefunctions,
        # else self.setup.objectivefunction
        self.objectivefunction = getattr(
            objectivefunctions, alt_objfun or '', None) or self.setup.objectivefunction
        self.evaluation = self.setup.evaluation()
        self.save_sim = save_sim
        self.dbname = dbname
        self.dbformat = dbformat
        self.breakpoint = breakpoint
        self.backup_every_rep = backup_every_rep
        self.dbinit = dbinit
        
        if breakpoint == 'read' or breakpoint == 'readandwrite':
            print('Reading backupfile')
            self.dbinit = False
            self.breakdata = self.read_breakdata(self.dbname)
        #self.initialize_database()

        # Now a repeater (ForEach-object) is loaded
        # A repeater is a convinent wrapper to repeat tasks
        # We have the same interface for sequential and for parallel tasks
        if parallel == 'seq':
            from spotpy.parallel.sequential import ForEach
        elif parallel == 'mpi':
            from spotpy.parallel.mpi import ForEach
        elif parallel == 'mpc':
            print('Multiprocessing is in still testing phase and may result in errors')
            from spotpy.parallel.mproc import ForEach
            #raise NotImplementedError(
            #    'Sorry, mpc is not available by now. Please use seq or mpi')
        else:
            raise ValueError(
                "'%s' is not a valid keyword for parallel processing" % parallel)

        # This is the repeater for the model runs. The simulate method does the work
        # If you need different tasks, the repeater can be pushed into a "phase" using the
        # setphase function. The simulate method can check the current phase and dispatch work
        # to other functions. This is introduced for sceua to differentiate between burn in and
        # the normal work on the chains
        self.repeat = ForEach(self.simulate)

        # In MPI, this command will do nothing on the master process
        # but the worker processes are going to wait for jobs.
        # Hence the workers will only receive parameters for the
        # simulate function, new calculation phases and the termination
        self.repeat.start()
        self.status = _RunStatistic()

    def set_repetiton(self, repetitions):
        self.status.repetitions = repetitions
        
    def final_call(self):
        self.repeat.terminate()
        try:
            self.datawriter.finalize()
        except AttributeError:  # Happens if no database was assigned
            pass
        print('End of sampling')
        text = 'Best run at %i of %i (best like=%g) with parameter set:' % (
            self.status.bestrep, self.status.repetitions, self.status.objectivefunction)
        print(text)
        print(self.status.params)
        text = 'Duration:' + str(round((time.time() - self.status.starttime), 2)) + ' s'
        print(text)
    
    def save(self, like, randompar, simulations, chains=1):
        # Initialize the database if no run was performed so far
        if self.dbformat and self.status.rep == 0:
            print('Initialize database...')
            writerclass = getattr(database, self.dbformat)
            
            self.datawriter = writerclass(
                self.dbname, self.parnames, like, randompar, simulations, save_sim=self.save_sim, 
                dbinit=self.dbinit)
        else:
            self.datawriter.save(like, randompar, simulations, chains=chains)

    def read_breakdata(self, dbname):
        ''' Read data from a pickle file if a breakpoint is set.
            Reason: In case of incomplete optimizations, old data can be restored. 
        '''
        import pickle
        #import pprint
        with open(dbname+'.break', 'rb') as csvfile:
            return pickle.load(csvfile)
#            pprint.pprint(work)
#            pprint.pprint(r)
#            pprint.pprint(icall)
#            pprint.pprint(gnrg)
            # icall = 1000 #TODO:Just for testing purpose

    def write_breakdata(self, dbname, work):
        ''' Write data to a pickle file if a breakpoint has been set.
        '''
        import pickle
        with open(str(dbname)+'.break', 'wb') as csvfile:
            pickle.dump(work, csvfile)

    def getdata(self):
        if self.dbformat == 'ram':
            return self.datawriter.data
        if self.dbformat == 'csv':
            return np.genfromtxt(self.dbname + '.csv', delimiter=',', names=True)[1:]
        if self.dbformat == 'sql':
            return self.datawriter.getdata
        if self.dbformat == 'noData':
            return self.datawriter.getdata

    def postprocessing(self, rep, randompar, simulation, chains=1, save=True, negativlike=False):
        like = self.getfitness(simulation=simulation, params=randompar)
        # Save everything in the database, if save is True
        # This is needed as some algorithms just want to know the fitness,
        # before they actually save the run in a database (e.g. sce-ua)
        if save is True:
            if negativlike is True:
                self.save(-like, randompar, simulations=simulation, chains=chains)              
                self.status(rep, -like, randompar)
            else:
                self.save(like, randompar, simulations=simulation, chains=chains)
                self.status(rep, like, randompar)
        if type(like)==type([]):
            return like[0]
        else:        
            return like
    
    
    def getfitness(self, simulation, params):
        """
        Calls the user defined spot_setup objectivefunction
        """
        try:
            #print('Using parameters in fitness function')
            return self.objectivefunction(evaluation=self.evaluation, simulation=simulation, params = (params,self.parnames))

        except TypeError: # Happens if the user does not allow to pass parameter in the spot_setup.objectivefunction
            #print('Not using parameters in fitness function')            
            return self.objectivefunction(evaluation=self.evaluation, simulation=simulation)
    
    def simulate(self, id_params_tuple):
        """This is a simple wrapper of the model, returning the result together with
        the run id and the parameters. This is needed, because some parallel things
        can mix up the ordering of runs
        """
        id, params = id_params_tuple
        return id, params, self.model(params)
