import numpy as np
from spotpy.tools.fixedrandom import *
from . import _algorithm

class DDS(_algorithm):
    """
        http://www.civil.uwaterloo.ca/btolson/software.aspx
        Paper:
            Tolson, B. A. and C. A. Shoemaker (2007), Dynamically dimensioned search algorithm for computationally efficient watershed model calibration, Water Resources Research, 43, W01413, 10.1029/2005WR004723.
            Asadzadeh, M. and B. A. Tolson (2013), Pareto archived dynamically dimensioned search with hypervolume-based selection for multi-objective optimization, Engineering Optimization. 10.1080/0305215X.2012.748046.
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

        # self.np_random = np.random
        self.np_random = FixedRandomizer()

    def __set_np_random(self,f_rand):
        self.np_random = f_rand

    def sample(self, repetitions,obj_func,fraction1,trials = 1):
        """
            --- --- --- -- .--.--.-.--.---.---.-.-.-..-.-.--
        """

        result_list = []
        sinitial, its, to_max = [], np.max([5, round(0.005 * repetitions)]), 1

        #print(self.objectivefunction([121312,12,12],[1214123,34,34]))
        #exit()

        self.set_repetiton(repetitions)

        self.min_bound, self.max_bound = self.parameter()['minbound'], self.parameter()['maxbound']
        self.discrete_flag = 0  # TODO determine if variable is type : continuous (0) or integer (1)

        num_dec = len(self.min_bound)  # num_dec is the number of decision variables

        for trial in range(trials):

            solution = np.array(repetitions * [(3 + num_dec) * [0.0]])

            stest = []
            sbest = []
            Jbest = []

            s_range = self.max_bound-self.min_bound


            # =================================================================================================
            # INITIAL SOLUTION
            # =================================================================================================

            # own initial solution:
            # sinitial = self.parameter()['random']

            if its > 1:  # its is the number of function evaluations to initialize the DDS algorithm solution
                print('Finding best starting point for trial '+str(trial)+' using '+str(its)+' random samples.')
                ileft = repetitions - its  # use this to reduce number of fevals in DDS loop
                if ileft <= 0:
                    raise ValueError('# Initialization samples >= Max # function evaluations.')

                for i in range(its):
                    if self.discrete_flag == 0:  # continuous variable
                        stest = self.min_bound + s_range * self.np_random.rand(num_dec)

                    else:  # discrete case
                        for j in range(num_dec):
                            stest[j] = self.np_random.randint(self.min_bound[j], self.max_bound[j] + 1)

                    Jtest = to_max * obj_func(stest)  # get obj function value

                    if i == 0:
                        Jbest = Jtest

                    if Jtest <= Jbest:
                        Jbest = Jtest
                        sbest = list(stest)

                    solution[i, 0] = i
                    solution[i, 1] = to_max * Jbest
                    solution[i, 2] = to_max * Jtest
                    solution[i, 3:3 + num_dec] = stest

            else:  # know its=1, using a user supplied initial solution.  Calculate obj func value.
                ileft = repetitions - 1  # use this to reduce number of fevals in DDS loop
                stest = sinitial  # get from the inputs
                Jtest = get_objfunc(stest)  # get obj function value
                Jbest = Jtest
                sbest = list(stest)
                solution[0, 0] = 1
                solution[0, 1] = to_max * Jbest
                solution[0, 2] = to_max * Jtest
                solution[0, 3:3 + num_dec] = stest



            it_sbest = its  # needed to initialize variable and avoid code failure when small # iterations
            trial_initial = list(sbest)  # extra variable here to simplify code for tracking initial DDS solution


            #
            # # A generator that produces parametersets if called
            # param_generator = ((rep, self.parameter()['random'])
            #                    for rep in range(int(repetitions)))
            # for rep, randompar, simulations in self.repeat(param_generator):
            #     # A function that calculates the fitness of the run and the manages the database
            #     self.postprocessing(rep, randompar, simulations)
            # self.final_call()



            #param_generator = ((rep, 1.0 - np.log(rep + 1) / np.log(ileft), np_random.rand(num_dec)) for rep in range(int(ileft)))

            # TODO implement like this!
            # import pprint
            # pprint.pprint(list(param_generator))
            # exit()

            for i in range(ileft):  # remaining F evals after initialization
                # Determine variable selected as neighbour
                Pn = 1.0 - np.log(i + 1) / np.log(ileft)  # 1.0-i/ileft;# probability of being selected as neighbour
                dvn_count = 0  # counter for how many decision variables vary in neighbour
                stest = list(sbest)  # define stest initially as current (sbest for greedy)

                randnums = self.np_random.rand(num_dec)


                for j in range(num_dec):
                    if randnums[j] < Pn:  # then j th DV selected to vary in neighbour
                        dvn_count = dvn_count + 1
                        new_value = self.neigh_value_mixed(sbest[j], self.min_bound[j], self.max_bound[j], fraction1, j + 1)

                        # TODO make this method!!
                        # TODO more efficient!!

                        stest[j] = new_value  # change relevant dec var value in stest

                # print(choosed_nums)
                # print(stest)
                # print("--------------------")

                if dvn_count == 0:  # no DVs selected at random, so select ONE
                    # TODO back: dec_var = np.int(np.ceil((num_dec-1) * np.random.rand()))  # which dec var to modify for neighbour
                    dec_var = np.int(np.ceil((num_dec) * self.np_random.rand()))

                    new_value = self.neigh_value_mixed(sbest[dec_var - 1], self.min_bound[dec_var - 1], self.max_bound[dec_var - 1], fraction1,
                                                  dec_var - 1)
                    # TODO more efficient!

                    stest[dec_var - 1] = new_value  # change relevant dec var value in stest

                # get ojective function value

                Jtest = to_max * obj_func(stest)

                # print([Jtest, Jbest]);
                # print(stest)
                if Jtest <= Jbest:
                    Jbest = Jtest
                    sbest = list(stest)
                    it_sbest = i + its  # iteration number best solution found

                    ### write new status file so that best sol'n not lost with long
                    ### runs (i.e. SWAT or other models called).  June 05 - BT
                    # Comment this part of code out for fast problems!!
                    #   filenam='status.out';
                    #   fid = fopen(filenam,'w'); % opens file and discards current contents
                    #   zzz=to_max*Jbest;
                    #   fprintf(fid,'Current best objective function value of %12.5f found at iteration %6.0f\n',zzz,i+its);
                    #   fprintf(fid,'under parameter set below: \n');
                    #   fprintf(fid,' %e ',sbest);
                    #   fclose(fid);
                    ###

                # accumulate results
                solution[i + its, 0] = i + its
                solution[i + its, 1] = to_max * Jbest
                solution[i + its, 2] = to_max * Jtest
                solution[i + its, 3:3 + num_dec] = stest

            # end DDS function loop

            print('Best solution found has obj function value of ' + str(to_max * Jbest) + ' \n\n')
            # [list(solution), it_sbest, sbest, trial_initial]

            result_list.append({"sbest": sbest, "trial_initial": trial_initial, "objfunc_val": to_max * Jbest})
        return result_list


    def neigh_value_continuous(self,s, s_min, s_max, fraction1):
        # select a RANDOM neighbouring real value of a SINGLE decision variable
        # CEE 509, HW 5 by Bryan Tolson, Mar 5, 2003 AND ALSO CEE PROJECT

        # variables:
        # s is a current SINGLE decision variable VALUE
        # s_min is the min of variable s
        # s_max is the max of variable s
        # snew is the neighboring VALUE of the decision variable
        # fraction1 is the neighbourhood parameter (replaces V parameter~see notes)
        #           It is defined as the ratio of the std deviation of the desired
        #           normal random number/s_range.  Eg:
        #               std dev desired = fraction1 * s_range
        #               for comparison:  variance (V) = (fraction1 * s_range)^2
        # s_range is the range of the real variable (s_max-s_min)

        s_range = s_max - s_min

        snew = s + self.np_random.normal(0, 1) * fraction1 * s_range

        # NEED to deal with variable upper and lower bounds:
        # Originally bounds in DDS were 100# reflective
        # But some times DVs are right on the boundary and with 100# reflective
        # boundaries it is hard to detect them. Therefore, we decided to make the
        # boundaries reflective with 50# chance and absorptive with 50# chance.
        # M. Asadzadeh and B. Tolson Dec 2008

        P_Abs_or_Ref = self.np_random.rand()

        if snew < s_min:  # works for any pos or neg s_min
            if P_Abs_or_Ref <= 0.5:  # with 50%chance reflect
                snew = s_min + (s_min - snew)
            else:  # with 50% chance absorb
                snew = s_min

                # if reflection goes past s_max then value should be s_min since without reflection
                # the approach goes way past lower bound.  This keeps X close to lower bound when X current
                # is close to lower bound:
            if snew > s_max:
                snew = s_min


        elif snew > s_max:  # works for any pos or neg s_max
            if P_Abs_or_Ref <= 0.5:  # with 50% chance reflect
                snew = s_max - (snew - s_max)
            else:  # with 50% chance absorb
                snew = s_max

                # if reflection goes past s_min then value should be s_max for same reasons as above
            if snew < s_min:
                snew = s_max

        return snew

    def neigh_value_discrete(self,s, s_min, s_max, fraction1):
        # Created by B.Tolson and B.Yung, June 2006
        # Modified by B. Tolson & M. Asadzadeh, Sept 2008
        # Modification: 1- Boundary for reflection at (s_min-0.5) & (s_max+0.5)
        #               2- Round the new value at the end of generation.
        # select a RANDOM neighbouring integer value of a SINGLE decision variable
        # discrete distribution is approximately normal
        # alternative to this appoach is reflecting triangular distribution (see Azadeh work)

        # variables:
        # s is a current SINGLE decision variable VALUE
        # s_min is the min of variable s
        # s_max is the max of variable s
        # delta_s_min is the minimum perturbation size for each decision variable
        # equals [] if continuous DV (blank)
        # equals 1 if discrete integer valued DV
        # snew is the neighboring VALUE of the decision variable
        # fraction1 is the neighbourhood parameter (replaces V parameter~see notes)
        #           It is defined as the ratio of the std deviation of the desired
        #           normal random number/s_range.  Eg:
        #               std dev desired = fraction1 * s_range
        #               for comparison:  variance (V) = (fraction1 * s_range)^2

        # s_range is the range of the real variable (s_max-s_min)
        s_range = s_max - s_min
        delta = self.np_random.normal(0, 1) * fraction1 * s_range
        snew = s + delta

        P_Abs_or_Ref = self.np_random.rand()

        if snew < s_min - 0.5:  # works for any pos or neg s_min
            if P_Abs_or_Ref <= 0.5:  # with 50% chance reflect
                snew = (s_min - 0.5) + ((s_min - 0.5) - snew)
            else:  # with 50% chance absorb
                snew = s_min

                # if reflection goes past (s_max+0.5) then value should be s_min since without reflection
                # the approach goes way past lower bound.  This keeps X close to lower bound when X current
                # is close to lower bound:
                if snew > s_max + 0.5:
                    snew = s_min

        elif snew > s_max + 0.5:  # works for any pos or neg s_max
            if P_Abs_or_Ref <= 0.5:  # with 50% chance reflect
                snew = (s_max + 0.5) - (snew - (s_max + 0.5))
            else:  # with 50% chance absorb
                snew = s_max

                # if reflection goes past (s_min-0.5) then value should be s_max for same reasons as above
            if snew < s_min - 0.5:
                snew = s_max

        snew = np.round(snew)  # New value must be integer
        if snew == s:  # pick a number between s_max and s_min by a Uniform distribution
            sample = s_min - 1 + np.ceil((s_max - s_min) * self.np_random.rand())
            if sample < s:
                snew = sample
            else:  # must increment option number by one
                snew = sample + 1
        return snew

    def neigh_value_mixed(self,s, s_min, s_max, fraction1, j):
        if self.discrete_flag == 0:
            return self.neigh_value_continuous(s, s_min, s_max, fraction1)
        else:
            return self.neigh_value_discrete(s, s_min, s_max, fraction1)



    # TODO: getestet werden sollten alle 5 Ergebnisvektoren




