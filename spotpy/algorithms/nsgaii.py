'''
Copyright (c) 2018 by Benjamin Manns
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Benjamin Manns

This file contains the NSGA-II Algorithm implemented for SPOTPY based on:
- K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm:
NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, Apr 2002.
doi: 10.1109/4235.996017
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=996017&isnumber=21497
- http://www.cs.colostate.edu/~genitor/MiscPubs/tutorial.pdf (Mutation and Crossover Algorithm)
- http://www.tik.ee.ethz.ch/file/6c0e384dceb283cd4301339a895b72b8/TIK-Report11.pdf (Tournament Selection)
'''

import numpy as np

from spotpy.algorithms import _algorithm
from sys import exit


class ParaPop:
    def __init__(self, params, m_vals=[], sim=None):
        self.params = params
        self.m_vals = m_vals
        self.sim = sim

    def __str__(self):
        return "<ParaPop> with content " + str(self.m_vals)

    def __repr__(self):
        return self.__str__()


class NSGAII(_algorithm):
    """
        Implements the "Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II
        by Kalyanmoy Deb, Associate Member, IEEE, Amrit Pratap, Sameer Agarwal, and T. Meyarivan

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

        super(NSGAII, self).__init__(*args, **kwargs)
        # self.all_objectives = [self.setup.__getattribute__(m) for m in dir(self.setup) if "objectivefunc" in m]

        self.param_len = len(self.get_parameters())
        self.generation = 0
        self.length_of_objective_func = 0

        self.objfun_maxmin_list = None
        self.objfun_normalize_list= []

    def fast_non_dominated_sort(self, P):
        S = {}
        n = {}
        F = {}
        rank = {}
        F[1] = {}

        if self.objfun_maxmin_list is None:
            self.length_of_objective_func = 0
            # In the first iteration create this list to have it prepared for later use
            self.objfun_maxmin_list = []
            for _ in self.objectivefunction(self.setup.evaluation(), self.setup.simulation(P[0])):
                self.length_of_objective_func += 1
                self.objfun_maxmin_list.append([])

        param_generator = ((p, list(P[p])) for p in P)
        calculated_sims = list(self.repeat(param_generator))
        for p, par_p, sim_p in calculated_sims:

            S[p] = {}
            S_p_index = 0
            n[p] = 0
            for q, par_q, sim_q in calculated_sims:

                # check whether parameter set p or q is dominating so we test all objective functions here
                # https://cims.nyu.edu/~gn387/glp/lec1.pdf / Definition / Dominance Relation

                # m_diffs = np.array([])
                m_vals_p = np.array([])
                m_vals_q = np.array([])

                for i, m in enumerate(self.objectivefunction(self.setup.evaluation(), sim_q)):
                    m_vals_q = np.append(m_vals_q, m)
                    self.objfun_maxmin_list[i].append(m)

                for i, m in enumerate(self.objectivefunction(self.setup.evaluation(), sim_p)):
                    m_vals_p = np.append(m_vals_p, m)
                    self.objfun_maxmin_list[i].append(m)

                m_diffs = m_vals_q - m_vals_p

                # TODO ist Minimieren oder Maximieren richtig?

                pp_q = ParaPop(np.array(P[q]), list(m_vals_q.tolist()), sim_q)

                pp_p = ParaPop(np.array(P[q]), list(m_vals_p.tolist()), sim_p)

                # Allow here also more then 2
                # if p dominates q
                if (m_diffs >= 0).all and (m_diffs > 0).any():

                    S[p][S_p_index] = pp_q
                    S_p_index += 1

                # elif q dominates p:
                elif (m_diffs <= 0).all() and (m_diffs < 0).any():
                    # else:
                    n[p] += 1

            if n[p] == 0:
                rank[p] = 1
                F[1][p] = pp_p

        i = 1
        while len(F[i]) > 0:
            Q = {}
            # q_ind is just a useless indices which may has to change somehow, it is only for the dict we use
            q_ind = 0
            for p in F[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q[q_ind] = S[p][q]
                        q_ind += 1

            i += 1
            F[i] = Q

        return F

    def crowding_distance_assignement(self, I):
        l = len(I)

        if l > 2:

            I = list(I.values())

            ##################
            # I = sort(I,m)  #
            ##################

            sorting_m = []

            for i in I:
                sorting_m.append(i.m_vals)

            sorting_m = np.array(sorting_m)

            new_order = np.argsort(np.sqrt(np.sum(sorting_m ** 2, 1)))

            I = np.array(I)
            I = I[new_order]


            distance_I = list(np.repeat(0, l))
            distance_I[0] = distance_I[l - 1] = np.inf

            for_distance = []

            for i in I:
                for_distance.append(i.m_vals)

            which_obj_fn = 0
            for_distance = np.array(for_distance)
            for k in for_distance.T:
                tmp_dist = k[2:l] - k[0:l - 2]

                distance_I[1:l - 1] += tmp_dist / self.objfun_normalize_list[which_obj_fn]

                which_obj_fn += 1

            return distance_I

        else:
            return [np.inf]

    def sample(self, generations=2, paramsamp=20):
        self.repetitions = int(generations)
        self.status.repetitions = self.repetitions*paramsamp*2
        R_0 = {}

        for i in range(paramsamp * 2):
            R_0[i] = list(self.parameter()['random'])

        while self.generation < self.repetitions:

            F = self.fast_non_dominated_sort(R_0)

            print("GENERATION: " + str(self.generation) + " of " + str(generations))

            # Debuggin Issue
            # import matplotlib.pyplot as pl
            # from mpl_toolkits.mplot3d import Axes3D
            # layer = 0
            # fig = pl.figure()
            #
            # if self.length_of_objective_func == 2:
            #
            #     for i in F:
            #         if layer == 0:
            #             l_color = "b"
            #         else:
            #             l_color = "#"
            #             for _ in range(6):
            #                 l_color += ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"][
            #                     np.random.randint(16)]
            #         for j in F[i]:
            #             pl.plot(F[i][j].m_vals[0], F[i][j].m_vals[1], color=l_color, marker='o')
            #         layer += 1
            #
            #     pl.show()
            #
            # elif self.length_of_objective_func == 3:
            #
            #     ax = fig.add_subplot(111, projection='3d')
            #
            #     for i in F:
            #         for j in F[i]:
            #             ax.scatter(F[i][j].m_vals[0], F[i][j].m_vals[1],
            #                        F[i][j].m_vals[2])  # , lay_col[layer] + 'o')
            #         layer += 1
            #
            #     # ax.set_xlabel(m1_name)
            #     # ax.set_ylabel(m2_name)
            #     # ax.set_zlabel(m3_name)
            #
            #     pl.show()

            # Now sort again
            complete_sort_all_p = []

            # post-proccesing min-max values of each objective function:
            # reset the normalize list
            self.objfun_normalize_list = []
            for i in self.objfun_maxmin_list:
                # fill the normalize list
                self.objfun_normalize_list.append(abs(max(i)-min(i)))

            # reset the objfun_maxmin_list
            self.objfun_maxmin_list = None

            cChain = 0
            for k in F:

                # Save fronts we have now before sorting and mutation
                for o in F[k]:

                    self.postprocessing(self.generation, F[k][o].params, F[k][o].sim, cChain)

                F_I_distance = self.crowding_distance_assignement(F[k])

                # print(F_I_distance)

                # sort within the list and then add to general list
                # the partial order <_n is defined as follows:
                # i <_n j if(i_rank < j_rank) or (i_rank = j_rank
                # and i_distance > j_distance )
                i_distance_order = np.argsort(F_I_distance)

                F_ar = np.array(list(F[k].values()))
                if len(F[k]) > 0:
                    for a in F_ar[i_distance_order]:
                        complete_sort_all_p.append(a)

                    # TODO Why is algorithm to take all values again, I think only the first which are the best
                cChain +=1

            N = paramsamp
            complete_sort_all_p = np.array(complete_sort_all_p)
            P_new = complete_sort_all_p[0:N]
            Q_new = {}
            M = len(P_new)
            if M < N:
                P_new = np.append(P_new, complete_sort_all_p[0:N - M])
                if N > len(P_new):
                    exit("We still have to few parameters after selecting parent elements")

            list_index = 0
            while list_index < N:
                # select pairs... with tournament, because this is a sorted list we just use the
                # first occurence of a pick "out of M (= length of P_new)
                tmp_parm_1 = P_new[np.random.randint(0, M, 1)[0]].params
                tmp_parm_2 = P_new[np.random.randint(0, M, 1)[0]].params

                # select cross over point
                xover_point = np.random.randint(0, self.param_len - 1, 1)[0]

                # cross over
                A = np.append(tmp_parm_2[0:xover_point], tmp_parm_1[xover_point:])

                # mutation
                sign = [0, 1, -1][np.random.randint(0, 3, 1)[0]]
                Q_new[list_index] = A + sign * (A / 4)  # was 100 before

                list_index += 1

            self.generation += 1

            # merge P and Q
            for i in P_new.tolist():
                Q_new[list_index] = i.params
                list_index += 1
            R_0 = Q_new

        self.final_call()
