import numpy as np

from spotpy.algorithms.dds import DDSGenerator
from . import _algorithm
from spotpy.parameter import ParameterSet
from spotpy.pareto_tools import crowd_dist
from spotpy.pareto_tools import nd_check

def ZDT1(x):
    """
    This test function is used by Deb et al. 2002 IEEE to test NSGAII
    performance. There are 30 decision variables which are in [0,1].
    :param x:
    :return: Two Value Array
    """
    a = x[0] # objective 1 value
    g = 0
    for i in range(1,30):
        g = g + x[i]
    g = 1 + 9 * g / 29
    b = g * (1 - (x[0] / g) ** 0.5) # objective 2 value
    return np.array([a,b])



class padds(_algorithm):


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
        :param r: neighborhood size perturbation parameter (r) that defines the random perturbation size standard
                  deviation as a fraction of the decision variable range. Default is 0.2.
        :type r: float

        """

        try:
            self.r = kwargs.pop("r")
        except KeyError:
            self.r = 0.2  # default value

        super(padds, self).__init__(*args, **kwargs)

        self.np_random = np.random

        self.status.params = ParameterSet(self.parameter())

        self.dds_generator = DDSGenerator(self.np_random)
        # self.generator_repetitions will be set in `sample` and is needed to generate a
        # generator which sends back actual parameter s_test
        self.generator_repetitions = -1
        self.pareto_front = np.array([])
        self.dominance_flag = -2


    def _set_np_random(self, f_rand):
        self.np_random = f_rand
        self.dds_generator.np_random = f_rand

    def roulette_wheel(self, metric):
        cumul_metric = np.cumsum(metric)
        probability = self.np_random.rand() * cumul_metric[-1]
        levels = (cumul_metric >= probability)
        length = cumul_metric.shape[0]
        return np.array(range(length))[levels][0]

    def get_next_x_curr(self):
        """
        Fake a generator to run self.repeat to use multiprocessing
        """
        # We need to shift position and length of the sampling process
        for rep in range(self.generator_repetitions):
            #print("dominance_flag =", self.dominance_flag)
            if self.dominance_flag == -1:  # if the last generated solution was dominated
                #print("metric",self.metric)
                index = self.roulette_wheel(self.metric)
                self.sbest = self.pareto_front[index, self.num_objs:]
                self.Jbest = self.pareto_front[index,:self.num_objs]
            else: # otherwise use the last generated solution
                self.Jbest = self.Jtest
                self.sbest = self.stest


            #print("sbest", self.sbest)
            #print("Jbest", self.Jbest)
            self.status.objectivefunction = self.Jbest
            self.status.params = self.sbest
            self.fix_status_params_format()

            yield rep, self.calculate_next_s_test(self.status.params, rep, self.generator_repetitions, self.r)

    def sample(self, repetitions, trials=1, x_initial=np.array([])):

        # every iteration a map of all relevant values is stored, only for debug purpose.
        # Spotpy will not need this values.
        debug_results = []

        self.set_repetiton(repetitions)

        # TODO set multiple objectives
        self.num_objs = 2
        self.number_of_parameters = len(self.status.params) # number_of_parameters is the amount of parameters

        # TODO set over setup
        self.Jtest = np.array([0.0]*self.num_objs)
        self.stest = np.array([0.0]*self.number_of_parameters)
        self.parameter_range = self.status.params.maxbound - self.status.params.minbound
        self.pareto_front = np.array([np.append([np.inf]*self.num_objs, [0]*self.number_of_parameters)])

        if len(x_initial) == 0:
            initial_iterations = np.int(np.max([5, round(0.005 * repetitions)]))
            self.calc_initial_pareto_front(initial_iterations)
        elif len(x_initial) != self.number_of_parameters:
            raise ValueError("User specified 'x_initial' has not the same length as available parameters")
        else:
            if not (np.all(x_initial <= self.status.params.maxbound) and np.all(
                    x_initial >= self.status.params.minbound)):
                raise ValueError("User specified 'x_initial' but the values are not within the parameter range")
            initial_iterations = 0
            for i in range(x_initial.shape[0]):
                if x_initial.shape[1] == self.num_objs + self.number_of_parameters:

                    self.Jtest = x_initial[i,:self.num_objs]
                    self.stest = x_initial[i,self.num_objs:]
                else:
                    self.stest = x_initial[i]
                    self.Jtest = self.sample_multi_objective(self.stest)

                    # TODO Bad way to set the same value every loop!
                    initial_iterations = x_initial.shape[0]

                if i == 0: # Initial value
                    self.pareto_front = np.array([np.append(self.Jtest, self.stest)])
                    dominance_flag = 1
                else:
                    self.pareto_front, dominance_flag = nd_check(self.pareto_front, self.Jtest, self.stest)
                    if dominance_flag > -1:
                        print("Parento Front has been changed")
                self.dominance_flag = dominance_flag

        self.status.params = self.stest
        self.fix_status_params_format()


        # Users can define trial runs in within "repetition" times the algorithm will be executed
        for trial in range(trials):
            #self.status.objectivefunction = [-1e308] * self.num_objs
            self.status.objectivefunction = 1e-308
            # repitionno_best saves on which iteration the best parameter configuration has been found
            repitionno_best = initial_iterations  # needed to initialize variable and avoid code failure when small # iterations
            #repetions_left = self.calc_initial_para_configuration(initial_iterations, trial,repetitions, x_initial)
            repetions_left =  repetitions - initial_iterations

            self.fix_status_params_format()
            trial_best_value = self.status.params.copy()

            # Main Loop of PA-DDS
            self.metric = self.calc_metric()

            #print(self.status.params)

            # important to set this field `generator_repetitions` so that
            # method `get_next_s_test` can generate exact parameters
            self.generator_repetitions = repetions_left

            for rep, x_curr, simulations in self.repeat(self.get_next_x_curr()):
                #print("OUTPUT", list(x_curr))

                self.Jtest = self.sample_multi_objective(x_curr)

                if rep +1 == 228:
                    #print("dominance_flag", self.dominance_flag)
                    #print(rep + 1, self.pareto_front)
                    #print("________________")
                    #print("x_curr",x_curr)
                    #print("________________")
                    #print("Jtest", self.Jtest)

                    pass

                num_imp = np.sum(self.Jtest <= self.Jbest)
                num_deg = np.sum(self.Jtest > self.Jbest)

                if num_imp == 0 and num_deg > 0: # New solution is dominated by its parent
                    self.dominance_flag = -1
                else: # Do dominance check only if new solution is not diminated by its parent
                    self.pareto_front, self.dominance_flag = nd_check(self.pareto_front , self.Jtest, x_curr)
                    #print("new pareto front", self.pareto_front)
                    if self.dominance_flag != -1:
                        self.metric = self.calc_metric()

                    # matlab code calls it stest

                if rep + 1 == 228:
                    #print("special______")
                    #print(self.dominance_flag)
                    #print(num_imp)
                    #print(num_deg)
                    #print("pareto_front", self.pareto_front)
                    #exit(42)
                    pass

                # todo muss in postprocessing rein aber hat eine andere multifunctionale objective function
                self.status.params = x_curr
                self.stest = x_curr
                #print("strange",self.status.params)
                self.fix_status_params_format()
                #print("strange 2", self.status.params)
                #self.postprocessing(rep, x_curr, simulations, chains=trial)
                #self.fix_status_params_format()


            print('Best solution found has obj function value of ' + str(self.status.objectivefunction) + ' at '
                  + str(repitionno_best) + '\n\n')
            debug_results.append({"sbest": self.status.params, "objfunc_val": self.status.objectivefunction})
        self.final_call()
        print("pareto_fron", self.pareto_front)
        print("sbest", self.stest)
        return debug_results

    def calc_metric(self):
        # TODO check which metric was chosen
        # TODO implement Crowded Distance
        # TODO use crowded Dustance, already implemented
        return np.array([1]*self.pareto_front.shape[0])

    def calc_initial_pareto_front(self, its):
        dominance_flag = -1
        for i in range(its):
            # TODO Use vector calc
            for j in range(self.number_of_parameters):
                if self.status.params.as_int[j]:
                    self.stest[j] = self.np_random.randint(self.status.params.minbound[j], self.status.params.maxbound[j])
                else:
                    self.stest[j] = self.status.params.minbound[j] + self.parameter_range[j] * self.np_random.rand()  # uniform random

            self.Jtest = self.sample_multi_objective(self.stest)
            # First value will be used to initialize the values
            if i == 0:
                self.pareto_front = np.vstack([self.pareto_front, np.append(self.Jtest, self.stest)])
            else:
                (self.pareto_front, dominance_flag) = nd_check(self.pareto_front, self.Jtest, self.stest)

        self.dominance_flag = dominance_flag

    def fix_status_params_format(self):
        start_params = ParameterSet(self.parameter())
        start_params.set_by_array([j for j in self.status.params])
        self.status.params = start_params


    def sample_multi_objective(self, data):
        # TODO but in setup and be sure that is is all in maximizing cause algorithm minimize unil now!
        return ZDT1(data)
        #return np.array([-1*np.mean(data), -1*np.var(data)])


    def calculate_next_s_test(self, previous_x_curr, rep, rep_limit, r):
        """
        Needs to run inside `sample` method. Calculate the next set of parameters based on a given set.
        This is greedy algorithm belonging to the DDS algorithm.

        `probability_neighborhood` is a threshold at which level a parameter is added to neighbourhood calculation.

        Using a normal distribution
        The decision variable

        `dvn_count` counts how many parameter configuration has been exchanged with neighbourhood values.
        If no parameters has been exchanged just one will select and exchanged with it's neighbourhood value.

        :param previous_x_curr: A set of parameters
        :param rep: Position in DDS loop
        :param r: neighbourhood size perturbation parameter
        :return: next parameter set
        """
        amount_params = len(previous_x_curr)
        #print("INPUT", list(previous_x_curr))
        new_x_curr = previous_x_curr.copy()  # define new_x_curr initially as current (previous_x_curr for greedy)

        randompar = self.np_random.rand(amount_params)

        probability_neighborhood = 1.0 - np.log(rep + 1) / np.log(rep_limit)
        dvn_count = 0  # counter for how many decision variables vary in neighbour

        for j in range(amount_params):
            if randompar[j] < probability_neighborhood:  # then j th DV selected to vary in neighbour
                dvn_count = dvn_count + 1
                new_value = self.dds_generator.neigh_value_mixed(previous_x_curr, r, j)
                new_x_curr[j] = new_value  # change relevant dec var value in x_curr

        if dvn_count == 0:  # no DVs selected at random, so select ONE
            dec_var = np.int(np.ceil(amount_params * self.np_random.rand()))
            new_value = self.dds_generator.neigh_value_mixed(previous_x_curr, r, dec_var - 1)

            new_x_curr[dec_var - 1] = new_value  # change relevant decision variable value in s_test

        return new_x_curr
