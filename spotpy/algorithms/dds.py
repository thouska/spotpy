import numpy as np
from . import _algorithm
from spotpy.parameter import ParameterSet


class DDSGenerator:
    """
        This class is used by the DDS algorithm to generate a new sample of parameters based on the current one.
        Current parameter are exchanged in `ParameterSet` objects.
    """

    def __init__(self, np_random):
        self.np_random = np_random

    def neigh_value_continuous(self, s, x_min, x_max, r):
        """
        select a RANDOM neighbouring real value of a SINGLE decision variable
        CEE 509, HW 5 by Bryan Tolson, Mar 5, 2003 AND ALSO CEE PROJECT
        variables:
        x_range is the range of the real variable (s_max-s_min)

        :param s: is a current SINGLE decision variable VALUE
        :param x_min: is the min of variable s
        :param x_max: is the max of variable s
        :param r: is the neighbourhood parameter (replaces V parameter~see not
                             It is defined as the ratio of the std deviation of the desired
                            normal random number/x_range.  Eg:
                        std dev desired = r * x_range
                        for comparison:  variance (V) = (r * x_range)^2
        :return: x_new, a new sample of values in beetween a given range
        """

        x_range = x_max - x_min

        x_new = s + self.np_random.normal(0, 1) * r * x_range

        # NEED to deal with variable upper and lower bounds:
        # Originally bounds in DDS were 100# reflective
        # But some times DVs are right on the boundary and with 100# reflective
        # boundaries it is hard to detect them. Therefore, we decided to make the
        # boundaries reflective with 50# chance and absorptive with 50# chance.
        # M. Asadzadeh and B. Tolson Dec 2008

        p_abs_or_ref = self.np_random.rand()

        if x_new < x_min:  # works for any pos or neg x_min
            if p_abs_or_ref <= 0.5:  # with 50%chance reflect
                x_new = x_min + (x_min - x_new)
            else:  # with 50% chance absorb
                x_new = x_min

                # if reflection goes past x_max then value should be x_min since without reflection
                # the approach goes way past lower bound.  This keeps X close to lower bound when X current
                # is close to lower bound:
            if x_new > x_max:
                x_new = x_min

        elif x_new > x_max:  # works for any pos or neg x_max
            if p_abs_or_ref <= 0.5:  # with 50% chance reflect
                x_new = x_max - (x_new - x_max)
            else:  # with 50% chance absorb
                x_new = x_max

                # if reflection goes past x_min then value should be x_max for same reasons as above
            if x_new < x_min:
                x_new = x_max

        return x_new

    def neigh_value_discrete(self, s, s_min, s_max, r):
        """
        Created by B.Tolson and B.Yung, June 2006
        Modified by B. Tolson & M. Asadzadeh, Sept 2008
        Modification: 1- Boundary for reflection at (s_min-0.5) & (s_max+0.5)
                      2- Round the new value at the end of generation.
        select a RANDOM neighbouring integer value of a SINGLE decision variable
        discrete distribution is approximately normal
        alternative to this appoach is reflecting triangular distribution (see Azadeh work)

        :param s: is a current SINGLE decision variable VALUE
        :param s_min: is the min of variable s
        :param s_max: is the max of variable s
        :param r: r is the neighbourhood parameter (replaces V parameter~see notes)
                  It is defined as the ratio of the std deviation of the desired
                  normal random number/s_range.  Eg:
                      std dev desired = r * s_range
                      for comparison:  variance (V) = (r * s_range)^2
        :return: s_new, a new sample of values in beetween a given range
        """

        s_range = s_max - s_min
        delta = self.np_random.normal(0, 1) * r * s_range
        s_new = s + delta

        p_abs_or_ref = self.np_random.rand()

        if s_new < s_min - 0.5:  # works for any pos or neg s_min
            if p_abs_or_ref <= 0.5:  # with 50% chance reflect
                s_new = (s_min - 0.5) + ((s_min - 0.5) - s_new)
            else:  # with 50% chance absorb
                s_new = s_min

                # if reflection goes past (s_max+0.5) then value should be s_min since without reflection
                # the approach goes way past lower bound.  This keeps X close to lower bound when X current
                # is close to lower bound:
                if s_new > s_max + 0.5:
                    s_new = s_min

        elif s_new > s_max + 0.5:  # works for any pos or neg s_max
            if p_abs_or_ref <= 0.5:  # with 50% chance reflect
                s_new = (s_max + 0.5) - (s_new - (s_max + 0.5))
            else:  # with 50% chance absorb
                s_new = s_max

                # if reflection goes past (s_min-0.5) then value should be s_max for same reasons as above
            if s_new < s_min - 0.5:
                s_new = s_max

        s_new = np.round(s_new)  # New value must be integer
        if s_new == s:  # pick a number between s_max and s_min by a Uniform distribution
            sample = s_min - 1 + np.ceil((s_max - s_min) * self.np_random.rand())
            if sample < s:
                s_new = sample
            else:  # must increment option number by one
                s_new = sample + 1
        return s_new

    def neigh_value_mixed(self, x_curr, r, j, x_min, x_max):
        """

        :param x_curr:
        :type x_curr: ParameterSet
        :param r:
        :param j:
        :return:
        """
        s = x_curr[j]

        if not x_curr.as_int[j]:
            return self.neigh_value_continuous(s, x_min, x_max, r)
        else:
            return self.neigh_value_discrete(s, x_min, x_max, r)


class dds(_algorithm):
    """
        Implements the Dynamically dimensioned search algorithm for computationally efficient watershed model
        calibration
        by
        Tolson, B. A. and C. A. Shoemaker (2007), Dynamically dimensioned search algorithm for computationally efficient
         watershed model calibration, Water Resources Research, 43, W01413, 10.1029/2005WR004723.
        Asadzadeh, M. and B. A. Tolson (2013), Pareto archived dynamically dimensioned search with hypervolume-based
        selection for multi-objective optimization, Engineering Optimization. 10.1080/0305215X.2012.748046.

        http://www.civil.uwaterloo.ca/btolson/software.aspx

        Method:
        "The DDS algorithm is a novel and simple stochastic single-solution based heuristic global search
        algorithm that was developed for the purpose of finding good global solutions
        (as opposed to globally optimal solutions) within a specified maximum function (or model) evaluation limit."
        (Page 3)

        The DDS algorithm is a simple greedy algorithm, always using the best solution (min or max) from the current
        point of view. This may not lead to the global optimization.

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
        :param r: neighborhood size perturbation parameter (r) that defines the random perturbation size standard
                  deviation as a fraction of the decision variable range. Default is 0.2.
        :type r: float

        """

        try:
            self.r = kwargs.pop("r")
        except KeyError:
            self.r = 0.2  # default value
        kwargs['optimization_direction'] = 'maximize'
        kwargs['algorithm_name'] = 'Dynamically Dimensioned Search (DDS) algorithm'
        super(dds, self).__init__(*args, **kwargs)

        self.np_random = np.random

        #self.status.params_max = ParameterSet(self.parameter())

        # self.generator_repetitions will be set in `sample` and is needed to generate a
        # generator which sends back actual parameter s_test
        self.generator_repetitions = -1

        self.dds_generator = DDSGenerator(self.np_random)

    def _set_np_random(self, f_rand):
        self.np_random = f_rand
        self.dds_generator.np_random = f_rand

    def get_next_x_curr(self):
        """
        Fake a generator to run self.repeat to use multiprocessing
        """
        # We need to shift position and length of the sampling process
        for rep in range(self.generator_repetitions):
            yield rep, self.calculate_next_s_test(self.params_max, rep, self.generator_repetitions, self.r)

    def sample(self, repetitions, trials=1, x_initial=np.array([])):
        """
        Samples from the DDS Algorithm.

        DDS is a greedy type of algorithm since the current solution, also the best solution identified so far,
        is never updated with a solution that has an inferior value of the objective function.

        That means in detail:
        The DDS Algorithm starts with an initial phase:
        If the user does not defines an own initial configuration The DDS algorithm start with searching a parameter
        configuration in between the given parameter bounds.

        The next phase is the dds algorithm itself which runs in a loop `repetion` times:
        Based on the parameter configuration x_new the algorithm run the model and simulation with the current parameter set
        and calculates the objective function value called F_curr.

        If F_curr > F_best, where F_best is the current max value objective function value, we set x_best = x_curr and
        F_best = F_curr.

        Select k of all parameters to include them in the neighborhood calculation. This is performed by calculating a
        threshold probability_neighborhood (probability in neighbourhood).

        The neighbourhood calculation perturb x_best on standard normal distribution and reflect the result if it
        breaks the parameter boundary.
        The updated parameter configuration is called x_curr

        :param repetitions:  Maximum number of runs.
        :type repetitions: int
        :param trials: amount of runs DDS algorithm will be performed
        :param x_initial: set an initial trial set as a first parameter configuration. If the set is empty the algorithm
                         select an own initial parameter configuration
        :return: a key-value set of all parameter combination which has been used. May changed in future.
        """

        # every iteration a map of all relevant values is stored, only for debug purpose.
        # Spotpy will not need this values.
        debug_results = []

        self.set_repetiton(repetitions)
        self.min_bound, self.max_bound = self.parameter(
        )['minbound'], self.parameter()['maxbound']
        print('Starting the DDS algotrithm with '+str(repetitions)+ ' repetitions...')

        number_of_parameters = self.status.parameters  # number_of_parameters is the amount of parameters

        if len(x_initial) == 0:
            initial_iterations = np.int(np.max([5, round(0.005 * repetitions)]))
        elif len(x_initial) != number_of_parameters:
            raise ValueError("User specified 'x_initial' has not the same length as available parameters")
        else:
            initial_iterations = 1
            x_initial = np.array(x_initial)
            if not (np.all(x_initial <= self.max_bound) and np.all(
                    x_initial >= self.min_bound)):
                raise ValueError("User specified 'x_initial' but the values are not within the parameter range")

        # Users can define trial runs in within "repetition" times the algorithm will be executed
        for trial in range(trials):
            #objectivefunction_max = -1e308
            params_max = x_initial
            # repitionno_best saves on which iteration the best parameter configuration has been found
            repitionno_best = initial_iterations  # needed to initialize variable and avoid code failure when small # iterations
            params_max, repetions_left, objectivefunction_max = self.calc_initial_para_configuration(initial_iterations, trial,
                                                                                    repetitions, x_initial)
            params_max = self.fix_status_params_format(params_max)
            trial_best_value = list(params_max)#self.status.params_max.copy()
            
            # important to set this field `generator_repetitions` so that
            # method `get_next_s_test` can generate exact parameters
            self.generator_repetitions = repetions_left
            self.params_max = params_max
            for rep, x_curr, simulations in self.repeat(self.get_next_x_curr()):

                like = self.postprocessing(rep, x_curr, simulations, chains=trial)
                if like > objectivefunction_max:
                    objectivefunction_max = like
                    self.params_max = list(x_curr)
                    self.params_max = self.fix_status_params_format(self.params_max)

            print('Best solution found has obj function value of ' + str(objectivefunction_max) + ' at '
                  + str(repitionno_best) + '\n\n')
            debug_results.append({"sbest": self.params_max, "trial_initial": trial_best_value,"objfunc_val": objectivefunction_max})
        self.final_call()
        return debug_results

    def fix_status_params_format(self, params_max):
        start_params = ParameterSet(self.parameter())
        start_params.set_by_array([j for j in params_max])
        return start_params

    def calc_initial_para_configuration(self, initial_iterations, trial, repetitions, x_initial):
        #max_bound, min_bound = self.status.params_max.maxbound, self.status.params_max.minbound
        parameter_bound_range = self.max_bound - self.min_bound
        number_of_parameters = len(parameter_bound_range)
        discrete_flag = ParameterSet(self.parameter()).as_int
        params_max = x_initial
        objectivefunction_max = -1e308
        # Calculate the initial Solution, if `initial_iterations` > 1 otherwise the user defined a own one.
        # If we need to find an initial solution we iterating initial_iterations times to warm um the algorithm
        # by trying which randomized generated input matches best
        # initial_iterations is the number of function evaluations to initialize the DDS algorithm solution
        if initial_iterations > 1:
            print('Finding best starting point for trial ' + str(trial + 1) + ' using ' + str(
                initial_iterations) + ' random samples.')
            repetions_left = repetitions - initial_iterations  # use this to reduce number of fevals in DDS loop
            if repetions_left <= 0:
                raise ValueError('# Initialization samples >= Max # function evaluations.')

            starting_generator = (
                (rep, [self.np_random.randint(np.int(self.min_bound[j]), np.int(self.max_bound[j]) + 1) if
                       discrete_flag[j] else self.min_bound[j] + parameter_bound_range[j] * self.np_random.rand()
                       for j in
                       range(number_of_parameters)]) for rep in range(int(initial_iterations)))

            for rep, x_curr, simulations in self.repeat(starting_generator):
                like = self.postprocessing(rep, x_curr, simulations)  # get obj function value
                # status setting update
                if like > objectivefunction_max:
                    objectivefunction_max = like
                    params_max = list(x_curr)         
                    params_max = self.fix_status_params_format(params_max)

        else:  # now initial_iterations=1, using a user supplied initial solution.  Calculate obj func value.
            repetions_left = repetitions - 1  # use this to reduce number of fevals in DDS loop
            rep, x_test_param, simulations = self.simulate((0, x_initial))  # get from the inputs
            like = self.postprocessing(rep, x_test_param, simulations)
            if like > objectivefunction_max:
                    objectivefunction_max = like
                    params_max = list(x_test_param)
                    params_max = self.fix_status_params_format(params_max)
        return params_max, repetions_left, objectivefunction_max

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
        new_x_curr = previous_x_curr.copy()  # define new_x_curr initially as current (previous_x_curr for greedy)

        randompar = self.np_random.rand(amount_params)
        probability_neighborhood = 1.0 - np.log(rep + 1) / np.log(rep_limit)
        dvn_count = 0  # counter for how many decision variables vary in neighbour

        for j in range(amount_params):
            if randompar[j] < probability_neighborhood:  # then j th DV selected to vary in neighbour
                dvn_count = dvn_count + 1
                new_value = self.dds_generator.neigh_value_mixed(previous_x_curr, r, j, self.min_bound[j],self.max_bound[j])
                new_x_curr[j] = new_value  # change relevant dec var value in x_curr

        if dvn_count == 0:  # no DVs selected at random, so select ONE
            dec_var = np.int(np.ceil(amount_params * self.np_random.rand()))
            new_value = self.dds_generator.neigh_value_mixed(previous_x_curr, r, dec_var - 1, self.min_bound[j],self.max_bound[j])

            new_x_curr[dec_var - 1] = new_value  # change relevant decision variable value in s_test

        return new_x_curr
