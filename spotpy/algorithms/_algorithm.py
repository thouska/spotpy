'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska

This file holds the standards for every algorithm.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from spotpy import database, objectivefunctions
from spotpy import parameter
import numpy as np
import time
import threading

try:
    from queue import Queue
except ImportError:
    # If the running python version is 2.* we have only Queue available as a multiprocessing class
    # we need to stop the whole main process which this sleep for one microsecond otherwise the subprocess is not
    # finished and the main process can not access it and put it as garbage away (Garbage collectors cause)
    # However this slows down the whole simulation process and is a boring bug. Python3.x does not need this
    # workaround
    from Queue import Queue



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
    save_thresholde: float or list
        Compares the given value/list of values with return value/list of values from spot_setup.objectivefunction.
        If the objectivefunction value is higher, the results are saved in the database. If not they are ignored (saves storage).
    db_precision:np.float type
        set np.float16, np.float32 or np.float64 for rounding of floats in the output database
        Default is np.float16
    alt_objfun: str or None, default: 'rmse'
        alternative objectivefunction to be used for algorithm
        * None: the objfun defined in spot_setup.objectivefunction is used
        * any str: if str is found in spotpy.objectivefunctions, 
            this objectivefunction is used, else falls back to None 
            e.g.: 'log_p', 'rmse', 'bias', 'kge' etc.
    sim_timeout: float, int or None, default: None
        the defined model given in the spot_setup class can be controlled to break after 'sim_timeout' seconds if
        sim_timeout is not None.
        If the model run has been broken simlply '[nan]' will be returned.
    random_state: int or None, default: None
        the algorithms uses the number in random_state as seed for numpy. This way stochastic processes can be reproduced.
    """

    def __init__(self, spot_setup, dbname=None, dbformat=None, dbinit=True,
                 parallel='seq', save_sim=True, alt_objfun=None, breakpoint=None,
                 backup_every_rep=100, save_threshold=-np.inf, db_precision=np.float16,sim_timeout = None,
                 random_state=None):
        # Initialize the user defined setup class
        self.setup = spot_setup
        self.model = self.setup.simulation
        # Philipp: Changed from Tobi's version, now we are using both new class defined parameters
        # as well as the parameters function. The new method get_parameters
        # can deal with a missing parameters function
        #
        # For me (Philipp) it is totally unclear why all the samplers should call this function
        # again and again instead of
        # TODO: just storing a definite list of parameter objects here
        self.parameter = self.get_parameters
        self.parnames = self.parameter()['name']

        # Create a type to hold the parameter values using a namedtuple
        self.partype = parameter.get_namedtuple_from_paramnames(
            self.setup, self.parnames)

        # use alt_objfun if alt_objfun is defined in objectivefunctions,
        # else self.setup.objectivefunction
        self.objectivefunction = getattr(
            objectivefunctions, alt_objfun or '', None) or self.setup.objectivefunction
        self.evaluation = self.setup.evaluation()
        self.save_sim = save_sim
        self.dbname = dbname or 'customDb'
        self.dbformat = dbformat or 'custom'
        self.db_precision = db_precision
        self.breakpoint = breakpoint
        self.backup_every_rep = backup_every_rep
        self.dbinit = dbinit
        
        # Set the random state
        if random_state is None:
            random_state = np.random.randint(low=0, high=2**30)
        np.random.seed(random_state)

        # If value is not None a timeout will set so that the simulation will break after sim_timeout seconds without return a value
        self.sim_timeout = sim_timeout
        self.save_threshold = save_threshold

        if breakpoint == 'read' or breakpoint == 'readandwrite':
            print('Reading backupfile')
            self.dbinit = False
            self.breakdata = self.read_breakdata(self.dbname)

        # Now a repeater (ForEach-object) is loaded
        # A repeater is a convinent wrapper to repeat tasks
        # We have the same interface for sequential and for parallel tasks
        if parallel == 'seq':
            from spotpy.parallel.sequential import ForEach
        elif parallel == 'mpi':
            from spotpy.parallel.mpi import ForEach

        # MPC is based on pathos mutiprocessing and uses ordered map, so results are given back in the order
        # as the parameters are
        elif parallel == 'mpc':
            from spotpy.parallel.mproc import ForEach

        # UMPC is based on pathos mutiprocessing and uses unordered map, so results are given back in the order
        # as the subprocesses are finished which may speed up the whole simulation process but is not recommended if
        # objective functions do their calculation based on the order of the data because the order of the result is chaotic
        # and randomized
        elif parallel == 'umpc':
            from spotpy.parallel.umproc import ForEach
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

    def __str__(self):
        return '{type}({mtype}())->{dbname}'.format(
            type=type(self).__name__,
            mtype=type(self.setup).__name__,
            dbname=self.dbname)

    def __repr__(self):
        return '{type}()'.format(type=type(self).__name__)

    def get_parameters(self):
        """
        Returns the parameter array from the setup
        """
        return parameter.get_parameters_array(self.setup)

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

    def _init_database(self, like, randompar, simulations):
        if self.dbinit:
            print('Initialize database...')

            self.datawriter = database.get_datawriter(self.dbformat,
                self.dbname, self.parnames, like, randompar, simulations, save_sim=self.save_sim,
                dbinit=self.dbinit, db_precision=self.db_precision, setup=self.setup)

            self.dbinit = False

    def save(self, like, randompar, simulations, chains=1):
        # Initialize the database if no run was performed so far
        self._init_database(like, randompar, simulations)

        #try if like is a list of values compare it with save threshold setting
        try:
            if all(i > j for i, j in zip(like, self.save_threshold)): #Compares list/list
                self.datawriter.save(like, randompar, simulations, chains=chains)
        #If like value is not a iterable, it is assumed to be a float
        except TypeError: # This is also used if not threshold was set
            try:
                if like>self.save_threshold: #Compares float/float
                    self.datawriter.save(like, randompar, simulations, chains=chains)
            except TypeError:# float/list would result in an error, because it does not make sense
                if like[0]>self.save_threshold: #Compares list/float
                    self.datawriter.save(like, randompar, simulations, chains=chains)

    def read_breakdata(self, dbname):
        ''' Read data from a pickle file if a breakpoint is set.
            Reason: In case of incomplete optimizations, old data can be restored. '''
        import pickle
        with open(dbname+'.break', 'rb') as breakfile:
            return pickle.load(breakfile)

    def write_breakdata(self, dbname, work):
        ''' Write data to a pickle file if a breakpoint has been set.'''
        import pickle
        with open(str(dbname)+'.break', 'wb') as breakfile:
            pickle.dump(work, breakfile)

    def getdata(self):
        return self.datawriter.getdata()

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

        # we need a layer to fetch returned data from a threaded process into a queue.
        def model_layer(q,params):
            # Call self.model with a namedtuple instead of another sequence
            q.put(self.model(self.partype(*params)))

        # starting a queue, where in python2.7 this is a multiprocessing class and can cause errors because of
        # incompability which the main thread. Therefore only for older Python version a workaround follows
        que = Queue()
        sim_thread = threading.Thread(target=model_layer, args=(que, params))
        sim_thread.daemon = True
        sim_thread.start()


        # If self.sim_timeout is not None the self.model will break after self.sim_timeout seconds otherwise is runs as
        # long it needs to run
        sim_thread.join(self.sim_timeout)

        # If no result from the thread is given, i.e. the thread was killed from the watcher the default result is
        # '[nan]' otherwise get the result from the thread
        model_result = [np.NAN]
        if not que.empty():
            model_result = que.get()
        return id, params, model_result
