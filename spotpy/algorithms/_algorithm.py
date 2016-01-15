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
from spotpy import database
import numpy as np

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
        self.rep = None
        self.params = None
        self.objectivefunction = -1e308
    def __call__(self,rep,objectivefunction,params):
        if type(objectivefunction)==type([]):
            if objectivefunction[0]>self.objectivefunction:
                self.objectivefunction = objectivefunction[0]#Show only the first best objectivefunction when working with more than one objectivefunction
                self.params = params
                self.rep = rep
        else:
            if objectivefunction>self.objectivefunction:
                self.params = params
                self.objectivefunction = objectivefunction
                self.rep = rep
            return True
        return False
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
        
    """
    
    def __init__(self, spot_setup, dbname=None, dbformat=None, parallel='seq',save_sim=True):
        #Initialize the user defined setup class
        self.setup        = spot_setup
        self.model        = self.setup.simulation
        self.parameter    = self.setup.parameters
        self.objectivefunction   = self.setup.objectivefunction
        self.evaluation   = self.setup.evaluation()
        self.save_sim     = save_sim
        self.dbname       = dbname
        self.dbformat     = dbformat
        
        #Initialize the database with a first run
        randompar       = self.parameter()['random']
        parnames        = self.parameter()['name']
        simulations     = self.model(randompar)
        like            = self.objectivefunction(simulations,self.evaluation)
        if dbname is not None:
            writerclass     = getattr(database, self.dbformat)
            self.datawriter = writerclass(self.dbname,parnames,like,randompar,simulations,save_sim=self.save_sim)
        else:
            self.datawriter = spot_setup
            
        # Now a repeater (ForEach-object) is loaded
        # A repeater is a convinent wrapper to repeat tasks
        # We have the same interface for sequential and for parallel tasks
        if parallel == 'seq':
            from spotpy.parallel.sequential import ForEach
        elif parallel == 'mpi':            
            from spotpy.parallel.mpi import ForEach
        elif parallel == 'mpc':
            raise NotImplementedError('Sorry, mpc is not available by now. Please use seq or mpi')
        else:
            raise ValueError("'%s' is not a valid keyword for parallel processing" % parallel)
        
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
    
    def getdata(self):
        if self.dbformat=='ram':
            return self.datawriter.data
        if self.dbformat=='csv':
            return np.genfromtxt(self.dbname+'.csv',delimiter=',',names=True)[1:]
        
    def simulate(self,id_params_tuple):
        """This is a simple wrapper of the model, returning the result together with
        the run id and the parameters. This is needed, because some parallel things
        can mix up the ordering of runs
        """
        id,params = id_params_tuple
        return id,params,self.model(params)
