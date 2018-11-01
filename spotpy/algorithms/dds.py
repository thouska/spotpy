import numpy as np
from spotpy.tools.fixedrandom import *
from . import _algorithm


class DDS(_algorithm):
    """
        Implements the Dynamically dimensioned search algorithm for computationally efficient watershed model
        calibration
        by
        Tolson, B. A. and C. A. Shoemaker (2007), Dynamically dimensioned search algorithm for computationally efficient
         watershed model calibration, Water Resources Research, 43, W01413, 10.1029/2005WR004723.
        Asadzadeh, M. and B. A. Tolson (2013), Pareto archived dynamically dimensioned search with hypervolume-based
        selection for multi-objective optimization, Engineering Optimization. 10.1080/0305215X.2012.748046.

        http://www.civil.uwaterloo.ca/btolson/software.aspx
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

        super(DDS, self).__init__(*args, **kwargs)

        self.np_random = np.random
        self.min_bound, self.max_bound = self.parameter()['minbound'], self.parameter()['maxbound']

        if hasattr(self.setup, "params"):
            self.discrete_flag = [u.is_distinct for u in self.setup.params]
        else:
            self.discrete_flag = [False] * len(self.max_bound)

        # self.generator_repetitions will be set in `sample` and is needed to generate a generator which sends back actual parameter s_test
        self.generator_repetitions = -1

        # holds currents best parameter
        self.next_s_best = []

    def _set_np_random(self, f_rand):
        self.np_random = f_rand

    def get_next_s_test(self):
        for rep in range(self.generator_repetitions - 1):
            yield rep + 1, self.next_s_best

    def sample(self, repetitions, fraction1=0.2, trials=1, s_initial=[], to_max=1):
        """
        Samples from the DDS Algorithm. User can define an own 's_initial' parameter configuration set. If not `s_initial`
        is set, the algorithm defines an own

        :param repetitions:  Maximum number of runs.
        :type repetitions: int
        :param fraction1: value between 0 and 1
        :type fraction1: float
        :param trials: amount of runs DDS algorithm will be performed
        :param s_initial: set an initial trial set
        :param to_max: 1 to minimize objective function, -1 maximized objective function
        :return:
        """

        self.fraction1 = fraction1

        # Check if `to_max` is correct
        if to_max != 1 and to_max != -1:
            raise ValueError("please specify `to_max` as 1 or -1")

        result_list = []

        self.set_repetiton(repetitions)

        num_dec = len(self.min_bound)  # num_dec is the number of decision variables

        if len(s_initial) == 0:
            its = np.int(np.max([5, round(0.005 * repetitions)]))
        elif len(s_initial) != num_dec:
            raise ValueError("User specified 's_initial' has not the same length as available parameters")
        else:
            its = 1
            s_initial = np.array(s_initial)
            if not (np.all(s_initial <= self.max_bound) and np.all(s_initial >= self.min_bound)):
                raise ValueError("User specified 's_initial' but the values are not within the parameter range")

        # Users can define trial runs in within "repetition" times the algorithm will be executed
        for trial in range(trials):
            s_best = []
            j_best = []

            s_range = self.max_bound - self.min_bound

            # Calculate the initial Solution, if `its` > 1 otherwise the user defined a own one.
            # If we need to find an initial solution we iterating its times to warm um the algorithm by trying which
            # randomized generated input matches best (has minimal / maximum likelihood)

            if its > 1:  # its is the number of function evaluations to initialize the DDS algorithm solution
                print('Finding best starting point for trial ' + str(trial + 1) + ' using ' + str(
                    its) + ' random samples.')
                i_left = repetitions - its  # use this to reduce number of fevals in DDS loop
                if i_left <= 0:
                    raise ValueError('# Initialization samples >= Max # function evaluations.')

                starting_generator = (
                    (rep, [self.np_random.randint(np.int(self.min_bound[j]), np.int(self.max_bound[j]) + 1) if
                           self.discrete_flag[j] else self.min_bound[j] + s_range[j] * self.np_random.rand() for j in
                           range(int(num_dec))]) for rep in range(int(its)))

                for rep, s_test, simulations in self.repeat(starting_generator):
                    like = self.postprocessing(rep, s_test, simulations)  # get obj function value

                    j_test = to_max * like

                    if rep == 0:
                        j_best = j_test
                        s_best = list(s_test)

                    if j_test <= j_best:
                        j_best = j_test
                        s_best = list(s_test)

            else:  # now its=1, using a user supplied initial solution.  Calculate obj func value.
                i_left = repetitions - 1  # use this to reduce number of fevals in DDS loop
                s_test = list(s_initial)  # get from the inputs

                single_generator = ((i, s_test) for i in range(1))
                rep, s_test_param, simulations = next(self.repeat(single_generator))

                j_test = self.postprocessing(rep, s_test, simulations)

                j_best = j_test
                s_best = list(s_test)

            it_sbest = its  # needed to initialize variable and avoid code failure when small # iterations
            trial_initial = list(s_best)  # extra variable here to simplify code for tracking initial DDS solution
            self.next_s_best = list(s_best)

            # important to set this field `generator_repetitions` so that method `get_next_s_test` can generate exact paremters
            self.generator_repetitions = i_left

            self.next_s_best = self.calculate_next_s_test(num_dec, s_test, s_best, 0)

            for rep, s_test, simulations in self.repeat(self.get_next_s_test()):

                like = self.postprocessing(rep, s_test, simulations, chains=trial)

                j_test = to_max * like

                if j_test <= j_best:
                    j_best = j_test
                    s_best = list(s_test)
                    it_sbest = rep + its  # iteration number best solution found

                # end DDS function loop
                s_test = list(s_best)
                self.next_s_best = self.calculate_next_s_test(num_dec, s_test, s_best, rep)

            print('Best solution found has obj function value of ' + str(to_max * j_best) + ' \n\n')
            result_list.append({"sbest": s_best, "trial_initial": trial_initial, "objfunc_val": to_max * j_best})

        return result_list

    def calculate_next_s_test(self, num_dec, s_test, s_best, rep):
        randompar = self.np_random.rand(num_dec)

        Pn = 1.0 - np.log(rep + 1) / np.log(self.generator_repetitions)
        dvn_count = 0  # counter for how many decision variables vary in neighbour
        # s_test = list(s_best)  # define s_test initially as current (s_best for greedy)

        for j in range(num_dec):
            if randompar[j] < Pn:  # then j th DV selected to vary in neighbour
                dvn_count = dvn_count + 1

                new_value = self.neigh_value_mixed(s_best[j], self.min_bound[j], self.max_bound[j], self.fraction1,
                                                   j)
                s_test[j] = new_value  # change relevant dec var value in stest

        if dvn_count == 0:  # no DVs selected at random, so select ONE
            dec_var = np.int(np.ceil(num_dec * self.np_random.rand()))
            new_value = self.neigh_value_mixed(s_best[dec_var - 1], self.min_bound[dec_var - 1],
                                               self.max_bound[dec_var - 1], self.fraction1,
                                               dec_var - 1)

            s_test[dec_var - 1] = new_value  # change relevant dec var value in s_test

        return s_test

    def neigh_value_continuous(self, s, s_min, s_max, fraction1):
        """
        select a RANDOM neighbouring real value of a SINGLE decision variable
        CEE 509, HW 5 by Bryan Tolson, Mar 5, 2003 AND ALSO CEE PROJECT
        variables:
        s_range is the range of the real variable (s_max-s_min)

        :param s: is a current SINGLE decision variable VALUE
        :param s_min: is the min of variable s
        :param s_max: is the max of variable s
        :param fraction1: is the neighbourhood parameter (replaces V parameter~see not
                             It is defined as the ratio of the std deviation of the desired
                            normal random number/s_range.  Eg:
                        std dev desired = fraction1 * s_range
                        for comparison:  variance (V) = (fraction1 * s_range)^2
        :return:
        """

        s_range = s_max - s_min

        s_new = s + self.np_random.normal(0, 1) * fraction1 * s_range

        # NEED to deal with variable upper and lower bounds:
        # Originally bounds in DDS were 100# reflective
        # But some times DVs are right on the boundary and with 100# reflective
        # boundaries it is hard to detect them. Therefore, we decided to make the
        # boundaries reflective with 50# chance and absorptive with 50# chance.
        # M. Asadzadeh and B. Tolson Dec 2008

        p_abs_or_ref = self.np_random.rand()

        if s_new < s_min:  # works for any pos or neg s_min
            if p_abs_or_ref <= 0.5:  # with 50%chance reflect
                s_new = s_min + (s_min - s_new)
            else:  # with 50% chance absorb
                s_new = s_min

                # if reflection goes past s_max then value should be s_min since without reflection
                # the approach goes way past lower bound.  This keeps X close to lower bound when X current
                # is close to lower bound:
            if s_new > s_max:
                s_new = s_min

        elif s_new > s_max:  # works for any pos or neg s_max
            if p_abs_or_ref <= 0.5:  # with 50% chance reflect
                s_new = s_max - (s_new - s_max)
            else:  # with 50% chance absorb
                s_new = s_max

                # if reflection goes past s_min then value should be s_max for same reasons as above
            if s_new < s_min:
                s_new = s_max

        return s_new

    def neigh_value_discrete(self, s, s_min, s_max, fraction1):
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
        :param fraction1: fraction1 is the neighbourhood parameter (replaces V parameter~see notes)
                  It is defined as the ratio of the std deviation of the desired
                  normal random number/s_range.  Eg:
                      std dev desired = fraction1 * s_range
                      for comparison:  variance (V) = (fraction1 * s_range)^2
        :return:
        """

        s_range = s_max - s_min
        delta = self.np_random.normal(0, 1) * fraction1 * s_range
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

    def neigh_value_mixed(self, s, s_min, s_max, fraction1, j):
        if not self.discrete_flag[j]:
            return self.neigh_value_continuous(s, s_min, s_max, fraction1)
        else:
            return self.neigh_value_discrete(s, s_min, s_max, fraction1)
