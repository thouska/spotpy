import numpy as np
from spotpy.tools.fixedrandom import *
from . import _algorithm
from spotpy.examples.spot_setup_dds import ackley10

def neigh_value_continuous(s, s_min, s_max, fraction1):
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

    s_new = s + np.random.normal(0, 1) * fraction1 * s_range

    # NEED to deal with variable upper and lower bounds:
    # Originally bounds in DDS were 100# reflective
    # But some times DVs are right on the boundary and with 100# reflective
    # boundaries it is hard to detect them. Therefore, we decided to make the
    # boundaries reflective with 50# chance and absorptive with 50# chance.
    # M. Asadzadeh and B. Tolson Dec 2008

    p_abs_or_ref = np.random.rand()

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



class PDDSGen:
    def __init__(self, m_num_dec, rep ,m_s_min, m_s_max,m_r_val):
        self.rep = rep
        self.m_r_val = m_r_val
        self.m_num_dec = m_num_dec
        self.nslaves = 8 # TODO set this, i.e. nslaves = m_nprocessors - 1;
        self.m_s_min = m_s_min
        self.m_s_max = m_s_max


    def __iter__(self): #todo as parameter
        self.eval = 0
        self.m_sbest = []
        self.Cbest = [0.00] * self.m_num_dec
        self.ileft = self.rep-self.nslaves # todo check format, set this from outside
        self.m_Fbest = 0
        self.first_update = True
        self.state = "DDS_INIT_STATE" # DDS_INIT_STATE | DDS_SEARCH_STATE
        self.jct = 0
        self.ini_fevals = 0

        # todo set s_best

        return self

    def _set_p(self,p):
        self.p = p

    def is_ready_to_harvest(self): # todo update name
        return self.eval > self.nslaves

    def update_curr_best_solution(self,Ftest,stest):
        if Ftest <= self.m_Fbest or self.first_update:
            self.first_update = False
            self.m_Fbest = Ftest
            self.m_sbest = list(stest)


    def __next__(self):
        if self.eval > self.ileft + self.nslaves: # eval <= (ileft + nslaves)
            raise StopIteration

        #result = np.random.binomial(12,self.p,10)

        if self.state == "DDS_INIT_STATE":

            # TODO put this in the init
            if self.eval == 0:
                self.ini_fevals = np.int(np.max([5, round(0.005 * self.rep)]))
                if self.ini_fevals < self.nslaves:
                    self.ini_fevals = self.nslaves


            # todo if user supplies own initial solution

            self.m_stest = self.m_s_min + np.random.rand(self.m_num_dec) * (self.m_s_max - self.m_s_min)

            if self.eval == 0:
                self.m_sbest = list(self.m_stest)

            if self.eval == self.ini_fevals:
                self.state = "DDS_SEARCH_STATE"




        # Here we use the result comming from the ForEach
        elif self.state == "DDS_SEARCH_STATE":
            dvn_count = 0

            if self.eval <= 2*self.nslaves:
                Pn = 1.00 #have each slave perturb all parameters the first time through
            else:
                ileft = self.rep
                Pn = 1.0 - np.log(self.eval - 2 * self.nslaves)/np.log((ileft - 2 * self.nslaves)) # probability each DV selected

            print(Pn)

            self.m_stest = list(self.m_sbest)

            for j in range(self.m_num_dec):
                # Using ngd instead of idg=idg+1 should help when ORDERED is removed
                m_ngd = j + (self.eval - 1) * self.m_num_dec + self.jct
                # TODO ranval = m_harvest[m_ngd]; what the hack is that -> checkout DDS
                ranval = np.random.rand()

                # jth DV selected for perturbation
                if ranval < Pn:
                    dvn_count += 1
                    # call 1-D perturbation function to get new DV value (new_value):
                    new_value = neigh_value_continuous(self.m_sbest[j], self.m_s_min[j], self.m_s_max[j], self.m_r_val)
                    # note that r_val is the value of the DDS r perturbation size parameter (0.2 by default)
                    self.m_stest[j] = new_value # change relevant DV value in stest

            if dvn_count == 0:
                # TODO ranval = m_harvest[m_ngd]; what the hack is that -> checkout DDS
                ranval = np.random.rand()
                self.jct += 1
                dv = np.int(np.ceil(self.m_num_dec * ranval)-1) # 0-based index for one DV
                # call 1-D perturbation function to get new DV value (new_value):
                new_value = neigh_value_continuous(self.m_sbest[dv], self.m_s_min[dv], self.m_s_max[dv], self.m_r_val)
                self.m_stest[dv] = new_value # change relevant DV value in stest


            # Preparing work for slave -> transfer by next()
            # sending as a tuple

        self.eval += 1
        return self.m_stest, self.m_Fbest, self.Cbest




class PDDS(_algorithm):
    """
        Implements the pareto Dynamically dimensioned search algorithm for computationally efficient watershed model
        calibration with asynchron multi
        by
        Tolson, B. A. and C. A. Shoemaker (2007), Dynamically dimensioned search algorithm for computationally efficient
         watershed model calibration, Water Resources Research, 43, W01413, 10.1029/2005WR004723.
        Asadzadeh, M. and B. A. Tolson (2013), Pareto archived dynamically dimensioned search with hypervolume-based
        selection for multi-objective optimization, Engineering Optimization. 10.1080/0305215X.2012.748046.

        http://www.civil.uwaterloo.ca/btolson/software.aspx
    """

    def __init__(self, *args, **kwargs):
        super(PDDS, self).__init__(*args, **kwargs)

        self.np_random = np.random
        self.m_s_min, self.m_s_max = self.parameter()['minbound'], self.parameter()['maxbound']

        if hasattr(self.setup,"params"):
            self.discrete_flag = [u.is_distinct for u in self.setup.params]
        else:
            self.discrete_flag = [False] * len(self.max_bound)

        self.m_stest = []




    def _set_np_random(self, f_rand):
        self.np_random = f_rand

    def repeater_wrapper(self,tup):
        return ackley10(tup[0])

    def slave_worker(self):
        for i in range(self.rep):
            yield i , self.m_stest

    def _set_p(self, p):
        self.p = p

    def is_ready_to_harvest(self):  # todo update name
        return self.eval > self.nslaves

    def sample(self, repetitions, fraction1=0.2, trials=1, s_initial=[]):
        """
        Samples from the PDDS Algorithm
        :param repetitions:  Maximum number of runs.
        :type repetitions: int
        :param fraction1: value between 0 and 1
        :type fraction1: float
        :param trials: amount of runs DDS algorithm will be performed
        :param s_initial: set an initial trial set
        :return:
        """

        self.rep = repetitions
        self.m_r_val = fraction1
        self.m_num_dec = len(self.m_s_max)
        self.nslaves = 8  # TODO set this, i.e. nslaves = m_nprocessors - 1;


        self.eval = 0
        self.m_sbest = []
        self.Cbest = [0.00] * self.m_num_dec
        self.ileft = self.rep - self.nslaves  # todo check format, set this from outside
        self.m_Fbest = 0
        self.first_update = True
        self.state = "DDS_INIT_STATE"  # DDS_INIT_STATE | DDS_SEARCH_STATE
        self.jct = 0
        self.ini_fevals = 0

        # todo set s_best

        # todo if user supplies own initial solution --> here is the first afterwards second and rest

        self.m_stest = self.m_s_min + np.random.rand(self.m_num_dec) * (self.m_s_max - self.m_s_min)

        for repeated_value in self.repeat(self.slave_worker()): # maybe we have to change foreach

            # TODO strange fix
            print(repeated_value[0])
            rep, s_test, simulations = repeated_value[0]
            Ftest = self.postprocessing(rep, s_test, simulations)  # get obj function value

            if Ftest <= self.m_Fbest or self.first_update:
                self.first_update = False
                self.m_Fbest = Ftest
                self.m_sbest = list(s_test)




            if self.eval > self.ileft + self.nslaves:  # eval <= (ileft + nslaves)
                raise StopIteration

            # result = np.random.binomial(12,self.p,10)

            if self.state == "DDS_INIT_STATE":

                # TODO put this in the init
                if self.eval == 0:
                    self.ini_fevals = np.int(np.max([5, round(0.005 * self.rep)]))
                    if self.ini_fevals < self.nslaves:
                        self.ini_fevals = self.nslaves

                # todo if user supplies own initial solution

                self.m_stest = self.m_s_min + np.random.rand(self.m_num_dec) * (self.m_s_max - self.m_s_min)

                if self.eval == 0:
                    self.m_sbest = list(self.m_stest)

                if self.eval == self.ini_fevals:
                    self.state = "DDS_SEARCH_STATE"




            # Here we use the result comming from the ForEach
            elif self.state == "DDS_SEARCH_STATE":
                dvn_count = 0

                if self.eval <= 2 * self.nslaves:
                    Pn = 1.00  # have each slave perturb all parameters the first time through
                else:
                    ileft = self.rep
                    Pn = 1.0 - np.log(self.eval - 2 * self.nslaves) / np.log(
                        (ileft - 2 * self.nslaves))  # probability each DV selected

                print(Pn)

                self.m_stest = list(self.m_sbest)

                for j in range(self.m_num_dec):
                    # Using ngd instead of idg=idg+1 should help when ORDERED is removed
                    m_ngd = j + (self.eval - 1) * self.m_num_dec + self.jct
                    # TODO ranval = m_harvest[m_ngd]; what the hack is that -> checkout DDS
                    ranval = np.random.rand()

                    # jth DV selected for perturbation
                    if ranval < Pn:
                        dvn_count += 1
                        # call 1-D perturbation function to get new DV value (new_value):
                        new_value = neigh_value_continuous(self.m_sbest[j], self.m_s_min[j], self.m_s_max[j],
                                                           self.m_r_val)
                        # note that r_val is the value of the DDS r perturbation size parameter (0.2 by default)
                        self.m_stest[j] = new_value  # change relevant DV value in s_test

                if dvn_count == 0:
                    # TODO ranval = m_harvest[m_ngd]; what the hack is that -> checkout DDS
                    ranval = np.random.rand()
                    self.jct += 1
                    dv = np.int(np.ceil(self.m_num_dec * ranval) - 1)  # 0-based index for one DV
                    # call 1-D perturbation function to get new DV value (new_value):
                    new_value = neigh_value_continuous(self.m_sbest[dv], self.m_s_min[dv], self.m_s_max[dv],
                                                       self.m_r_val)
                    self.m_stest[dv] = new_value  # change relevant DV value in s_test

                # Preparing work for slave -> transfer by next()
                # sending as a tuple

            self.eval += 1
            # TODO use this for later?
            # return self.m_stest, self.m_Fbest, self.Cbest










































        #from spotpy.parallel.umproc import ForEach
        # from spotpy.parallel.sequential import ForEach
        #
        #
        # repeater = ForEach(self.repeater_wrapper)
        #
        # pddsgenerator = PDDSGen(len(self.m_s_min),repetitions,self.m_s_min,self.m_s_max,fraction1)
        # iter(pddsgenerator)
        #
        # for r in repeater(pddsgenerator):
        #     pddsgenerator.update_curr_best_solution(r[0],r[1][0])
        #     print("r,",r[0],r[1][0])

        # while True:
        #     print(next(pddsgenerator))