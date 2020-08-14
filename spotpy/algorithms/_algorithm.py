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
from spotpy import database
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

    def __init__(self, repetitions, algorithm_name, optimization_direction, parnames):
        self.optimization_direction = optimization_direction #grid, mazimize, minimize
        print('Initializing the ',algorithm_name,' with ',repetitions,' repetitions')
        if optimization_direction == 'minimize':
            self.compare = self.minimizer
            print('The objective function will be minimized')
        if optimization_direction == 'maximize':
            self.compare = self.maximizer
            print('The objective function will be maximized')
        if optimization_direction == 'grid':
            self.compare = self.grid

        self.rep = 0
        self.parnames = parnames
        self.parameters= len(parnames)
        self.params_min = [np.nan]*self.parameters
        self.params_max = [np.nan]*self.parameters
        self.objectivefunction_min = 1e308
        self.objectivefunction_max = -1e308
        self.starttime = time.time()
        self.last_print = time.time()
        
        self.repetitions = repetitions
        self.stop = False

    def minimizer(self, objval, params):
        if objval < self.objectivefunction_min:
            self.objectivefunction_min = objval
            self.params_min = list(params)

    def maximizer(self, objval, params):
        if objval > self.objectivefunction_max:
            self.objectivefunction_max = objval
            self.params_max = list(params)

    def grid(self, objval, params):
        if objval < self.objectivefunction_min:
            self.objectivefunction_min = objval
            self.params_min = list(params)
        if objval > self.objectivefunction_max:
            self.objectivefunction_max = objval
            self.params_max = list(params)


    def __call__(self, objectivefunction, params, block_print=False):
        self.rep+=1
        if type(objectivefunction) == type([]): #TODO: change to iterable
            self.compare(objectivefunction[0], params)
        elif type(objectivefunction) == type(np.array([])):
            pass
        else:
            self.compare(objectivefunction, params)


        if self.rep == self.repetitions:
            self.stop = True

        if not block_print:
            self.print_status()

    def print_status(self):
        # get str showing approximate timeleft to end of simulation in H, M, S
        acttime = time.time()
        # Refresh progressbar every two second
        if acttime - self.last_print >= 2:
            avg_time_per_run = (acttime - self.starttime) / (self.rep + 1)
            timestr = time.strftime("%H:%M:%S", time.gmtime(round(avg_time_per_run * (self.repetitions - (self.rep + 1)))))
            if self.optimization_direction == 'minimize':
                text = '%i of %i, minimal objective function=%g, time remaining: %s' % (
                        self.rep, self.repetitions, self.objectivefunction_min, timestr)

            if self.optimization_direction == 'maximize':
                text = '%i of %i, maximal objective function=%g, time remaining: %s' % (
                        self.rep, self.repetitions, self.objectivefunction_max, timestr)

            if self.optimization_direction == 'grid':
                text = '%i of %i, min objf=%g, max objf=%g, time remaining: %s' % (
                        self.rep, self.repetitions, self.objectivefunction_min, self.objectivefunction_max, timestr)

            print(text)
            self.last_print = time.time()

    def print_status_final(self):
        print('\n*** Final SPOTPY summary ***')
        print('Total Duration: ' + str(round((time.time() - self.starttime), 2)) + ' seconds')
        print('Total Repetitions:', self.rep)

        if self.optimization_direction == 'minimize':
            print('Minimal objective value: %g' % (self.objectivefunction_min))
            print('Corresponding parameter setting:')
            for i in range(self.parameters):
                text = '%s: %g' % (self.parnames[i], self.params_min[i])
                print(text)

        if self.optimization_direction == 'maximize':
            print('Maximal objective value: %g' % (self.objectivefunction_max))
            print('Corresponding parameter setting:')
            for i in range(self.parameters):
                text = '%s: %g' % (self.parnames[i], self.params_max[i])
                print(text)

        if self.optimization_direction == 'grid':
            print('Minimal objective value: %g' % (self.objectivefunction_min))
            print('Corresponding parameter setting:')
            for i in range(self.parameters):
                text = '%s: %g' % (self.parnames[i], self.params_min[i])
                print(text)

            print('Maximal objective value: %g' % (self.objectivefunction_max))
            print('Corresponding parameter setting:')
            for i in range(self.parameters):
                text = '%s: %g' % (self.parnames[i], self.params_max[i])
                print(text)

        print('******************************\n')


    def __repr__(self):
        return 'Min objectivefunction: %g \n Max objectivefunction: %g' % (
                self.objectivefunction_min, self.objectivefunction_max)


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
    save_threshold: float or list
        Compares the given value/list of values with return value/list of values from spot_setup.objectivefunction.
        If the objectivefunction value is higher, the results are saved in the database. If not they are ignored (saves storage).
    db_precision:np.float type
        set np.float16, np.float32 or np.float64 for rounding of floats in the output database
        Default is np.float16
    sim_timeout: float, int or None, default: None
        the defined model given in the spot_setup class can be controlled to break after 'sim_timeout' seconds if
        sim_timeout is not None.
        If the model run has been broken simlply '[nan]' will be returned.
    random_state: int or None, default: None
        the algorithms uses the number in random_state as seed for numpy. This way stochastic processes can be reproduced.
    """

    _unaccepted_parameter_types = (parameter.List, )

    def __init__(self, spot_setup, dbname=None, dbformat=None, dbinit=True,
                 dbappend=False, parallel='seq', save_sim=True, breakpoint=None,
                 backup_every_rep=100, save_threshold=-np.inf, db_precision=np.float16,
                 sim_timeout=None, random_state=None, optimization_direction='grid', algorithm_name=''):

        # Initialize the user defined setup class
        self.setup = spot_setup
        param_info = parameter.get_parameters_array(self.setup, unaccepted_parameter_types=self._unaccepted_parameter_types)
        self.all_params = param_info['random']
        self.constant_positions = parameter.get_constant_indices(spot_setup)
        if self.constant_positions:
            self.non_constant_positions = []
            for i, val in enumerate(self.all_params):
                if self.all_params[i] not in self.constant_positions:
                    self.non_constant_positions.append(i)
        else: 
            self.non_constant_positions = np.arange(0,len(self.all_params))
        self.parameter = self.get_parameters
        self.parnames = param_info['name']
        self.algorithm_name = algorithm_name
        # Create a type to hold the parameter values using a namedtuple
        self.partype = parameter.ParameterSet(param_info)

        self.evaluation = self.setup.evaluation()
        self.save_sim = save_sim
        self.optimization_direction = optimization_direction
        self.dbname = dbname or 'customDb'
        self.dbformat = dbformat or 'ram'
        self.db_precision = db_precision
        self.breakpoint = breakpoint
        self.backup_every_rep = backup_every_rep
        # Two parameters to control the data base handling
        # 'dbinit' triggers the initial creation of the data base file
        # 'dbappend' used to append to the existing data base, after restart
        self.dbinit = dbinit
        self.dbappend = dbappend
        
        # Set the random state
        if random_state is None: #ToDo: Have to discuss if these 3 lines are neccessary.
            random_state = np.random.randint(low=0, high=2**30)
        np.random.seed(random_state) 

        # If value is not None a timeout will set so that the simulation will break after sim_timeout seconds without return a value
        self.sim_timeout = sim_timeout
        self.save_threshold = save_threshold

        if breakpoint == 'read' or breakpoint == 'readandwrite':
            print('Reading backupfile')
            try:
                open(self.dbname+'.break')
            except FileNotFoundError:
                print('Backupfile not found')
            self.dbappend = True

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

        # method "save" needs to know whether objective function result is list or float, default is float
        self.like_struct_typ = type(1.1)
        
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
        pars = parameter.get_parameters_array(self.setup)
        return pars[self.non_constant_positions]

    def set_repetiton(self, repetitions):
        self.status = _RunStatistic(repetitions, self.algorithm_name,
                                    self.optimization_direction, self.parnames)
        # In MPI, this command will do nothing on the master process
        # but the worker processes are going to wait for jobs.
        # Hence the workers will only receive parameters for the
        # simulate function, new calculation phases and the termination
        self.repeat.start()

    def final_call(self):
        self.repeat.terminate()
        try:
            self.datawriter.finalize()
        except AttributeError:  # Happens if no database was assigned
            pass
        self.status.print_status_final()


    def _init_database(self, like, randompar, simulations):
        if self.dbinit:
            print('Initialize database...')

            self.datawriter = database.get_datawriter(self.dbformat,
                self.dbname, self.parnames, like, randompar, simulations,
                save_sim=self.save_sim, dbappend=self.dbappend,
                dbinit=self.dbinit, db_precision=self.db_precision,
                setup=self.setup)

            self.dbinit = False


    def __is_list_type(self, data):
        if type(data) == type:
            return data == list or data == type(np.array([]))
        else:
            return type(data) == list or type(data) == type(np.array([]))

    def save(self, like, randompar, simulations, chains=1):
        # Initialize the database if no run was performed so far
        self._init_database(like, randompar, simulations)
        # Test if like and the save threshold are float/list and compare accordingly
        if self.__is_list_type(like) and self.__is_list_type(self.save_threshold):
            if all(i > j for i, j in zip(like, self.save_threshold)): #Compares list/list
                self.datawriter.save(like, randompar, simulations, chains=chains)
        if (not self.__is_list_type(like)) and (not self.__is_list_type(self.save_threshold)):
            if like>self.save_threshold: #Compares float/float
                self.datawriter.save(like, randompar, simulations, chains=chains)
        if self.__is_list_type(like) and (not self.__is_list_type(self.save_threshold)):
            if like[0]>self.save_threshold: #Compares list/float
                self.datawriter.save(like, randompar, simulations, chains=chains)
        if (not self.__is_list_type(like)) and self.__is_list_type(self.save_threshold): #Compares float/list
            if (like > self.save_threshold).all:
                self.datawriter.save(like, randompar, simulations, chains=chains)

    def read_breakdata(self, dbname):
        ''' Read data from a pickle file if a breakpoint is set.
            Reason: In case of incomplete optimizations, old data can be restored. '''
        import pickle
        with open(dbname+'.break', 'rb') as breakfile:
            work,backuptime,repos,obmin,obmax=pickle.load(breakfile)
            self.status.starttime=self.status.starttime-backuptime
            self.status.rep=repos
            self.status.objectivefunction_min=obmin
            self.status.objectivefunction_max=obmax
            return work

    def write_breakdata(self, dbname, work):
        ''' Write data to a pickle file if a breakpoint has been set.'''
        import pickle
        work=(work,self.status.last_print-self.status.starttime,self.status.rep,self.status.objectivefunction_min,self.status.objectivefunction_max)
        with open(str(dbname)+'.break', 'wb') as breakfile:
            pickle.dump(work, breakfile)

    def getdata(self):
        return self.datawriter.getdata()

    def update_params(self, params):
        #Add potential Constant parameters
        self.all_params[self.non_constant_positions] = params
        return self.all_params
            
    
    def postprocessing(self, rep, params, simulation, chains=1, save_run=True, negativlike=False, block_print=False): # TODO: rep not necessaray
    
        params = self.update_params(params)
        if negativlike is True:
            like = -self.getfitness(simulation=simulation, params=params)
        else:
            like = self.getfitness(simulation=simulation, params=params)

        # Save everything in the database, if save is True
        # This is needed as some algorithms just want to know the fitness,
        # before they actually save the run in a database (e.g. sce-ua)

        self.status(like,params,block_print=block_print)
        
        if save_run is True and simulation is not None:
            self.save(like, params, simulations=simulation, chains=chains)
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
            return self.setup.objectivefunction(evaluation=self.evaluation, simulation=simulation, params = (params,self.parnames))

        except TypeError: # Happens if the user does not allow to pass parameter in the spot_setup.objectivefunction
            #print('Not using parameters in fitness function')            
            return self.setup.objectivefunction(evaluation=self.evaluation, simulation=simulation)
    
    def simulate(self, id_params_tuple):
        """This is a simple wrapper of the model, returning the result together with
        the run id and the parameters. This is needed, because some parallel things
        can mix up the ordering of runs
        """
        id, params = id_params_tuple
        self.all_params[self.non_constant_positions] = params #TODO: List parameters are not updated if not accepted for the algorithm, we may have to warn/error if list is given
        all_params = self.all_params

        # we need a layer to fetch returned data from a threaded process into a queue.
        def model_layer(q,all_params):
            # Call self.model with a namedtuple instead of another sequence
            q.put(self.setup.simulation(self.partype(*all_params)))

        # starting a queue, where in python2.7 this is a multiprocessing class and can cause errors because of
        # incompability which the main thread. Therefore only for older Python version a workaround follows
        que = Queue()
        sim_thread = threading.Thread(target=model_layer, args=(que, all_params))
        sim_thread.daemon = True
        sim_thread.start()


        # If self.sim_timeout is not None the self.model will break after self.sim_timeout seconds otherwise is runs as
        # long it needs to run
        sim_thread.join(self.sim_timeout)

        # If no result from the thread is given, i.e. the thread was killed from the watcher the default result is
        # '[nan]' and will not be saved. Otherwise get the result from the thread
        model_result = None
        if not que.empty():
            model_result = que.get()
        return id, params, model_result
