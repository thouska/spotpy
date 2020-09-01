import numpy as np
from copy import deepcopy

from spotpy.algorithms.dds import DDSGenerator
from . import _algorithm
from spotpy.parameter import ParameterSet
import copy
from scipy.spatial.qhull import ConvexHull, QhullError

class BestValue(object):
    """
        BestValue holds a parameter set and a best objective value, which is used by the PADDS Algorithm.
        Updates are done within the algorithm
    """

    def __init__(self, para_func, obj_value):
        self.para_func = para_func
        self.parameters = ParameterSet(self.para_func())
        self.best_obj_val = obj_value
        self.best_rep = 0


    def copy(self):
        to_copy = BestValue(self.parameters.copy(), self.best_obj_val)
        to_copy.best_rep = self.best_rep
        return to_copy

    def __str__(self):
        return "BestValue(best_obj_val = " + str(self.best_obj_val) + ", best_rep = " + str(self.best_rep) + ", " \
               + str(self.parameters) + ")"

    def reset_rep(self):
        self.best_rep = 0

    def fix_format(self):
        start_params = ParameterSet(self.para_func())
        start_params.set_by_array([j for j in self.parameters])
        self.parameters = start_params


class padds(_algorithm):
    """
    Implements the Pareto Archived Dynamically Dimensioned Search (short PADDS algorithm) by
    Tolson, B. A. and  Asadzadeh M. (2013)
    https://www.researchgate.net/publication/259982925_Pareto_archived_dynamically_dimensioned_search_with_hypervolume-based_selection_for_multi-objective_optimization

    PADDS using the DDS algorithm with a pareto front included. Two metrics are implemented,
    which is the simple "one" metric and the "crowd distance" metric.
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

        super(padds, self).__init__(*args, **kwargs)

        self.np_random = np.random


        self.best_value = BestValue(self.parameter, None)

        self.dds_generator = DDSGenerator(self.np_random)
        # self.generator_repetitions will be set in `sample` and is needed to generate a
        # generator which sends back actual parameter s_test
        self.generator_repetitions = -1
        self.pareto_front = np.array([])
        self.dominance_flag = -2
        self.obj_func_current = None
        self.parameter_current = None

        # because we have a pareto front we need another save type
        self.like_struct_typ = type([])

    def _set_np_random(self, f_rand):
        self.np_random = f_rand
        if hasattr(self,"hvc"):
            self.hvc._set_np_random(f_rand)
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
            if self.dominance_flag == -1:  # if the last generated solution was dominated
                index = self.roulette_wheel(self.metric)
                self.best_value.parameters, self.best_value.best_obj_val = self.pareto_front[index][1], self.pareto_front[index][0]
            else:  # otherwise use the last generated solution
                self.best_value.parameters, self.best_value.best_obj_val = (self.parameter_current, self.obj_func_current)

            # This line is needed to get an array of data converted into a parameter object
            self.best_value.fix_format()

            yield rep, self.calculate_next_s_test(self.best_value.parameters, rep, self.generator_repetitions, self.r)

    def calculate_initial_parameterset(self, repetitions, initial_objs, initial_params):
        self.obj_func_current = np.array([0.0])
        self.parameter_current = np.array([0.0] * self.number_of_parameters)
        self.parameter_range = self.best_value.parameters.maxbound - self.best_value.parameters.minbound
        self.pareto_front = np.array([[np.array([]), np.array([0] * self.number_of_parameters)]])
        #self.pareto_front = np.array([np.append([np.inf] * self.like_struct_len, [0] * self.number_of_parameters)])

        if(len(initial_objs) != len(initial_params)):
            raise ValueError("User specified 'initial_objs' and 'initial_params' have no equal length")

        if len(initial_objs) == 0:
            initial_iterations = np.int(np.max([5, round(0.005 * repetitions)]))
            self.calc_initial_pareto_front(initial_iterations)
        elif initial_params.shape[1] != self.number_of_parameters:
            raise ValueError("User specified 'initial_params' has not the same length as available parameters")
        else:
            if not (np.all(initial_params <= self.best_value.parameters.maxbound) and np.all(
                    initial_params >= self.best_value.parameters.minbound)):
                raise ValueError("User specified 'initial_params' but the values are not within the parameter range")
            initial_iterations = initial_params.shape[0]

            for i in range(initial_params.shape[0]):
                self.parameter_current = initial_params[i]
                if len(initial_objs[i]) > 0:
                    self.obj_func_current = initial_objs[i]
                else:
                    self.obj_func_current = self.getfitness(simulation=[], params=self.parameter_current)

                if i == 0:  # Initial value
                    self.pareto_front = np.array([[self.obj_func_current, self.parameter_current]])
                    dominance_flag = 1
                else:
                    self.pareto_front, dominance_flag = nd_check(self.pareto_front, self.obj_func_current,
                                                                 self.parameter_current.copy())
                self.dominance_flag = dominance_flag

        return initial_iterations, copy.deepcopy(self.parameter_current)

    def sample(self, repetitions, trials=1, initial_objs=np.array([]), initial_params=np.array([]), metric="ones"):
        # every iteration a map of all relevant values is stored, only for debug purpose.
        # Spotpy will not need this values.
        debug_results = []
        print('Starting the PADDS algotrithm with ' + str(repetitions) + ' repetitions...')
        print('WARNING: THE PADDS algorithm as implemented in SPOTPY is in an beta stage and not ready for production use!')
        self.set_repetiton(repetitions)
        self.number_of_parameters = len(self.best_value.parameters) # number_of_parameters is the amount of parameters

        if metric == "hvc":
            self.hvc = HVC(np_random=self.np_random)
        
        self.min_bound, self.max_bound = self.parameter()['minbound'], self.parameter()['maxbound']

        # Users can define trial runs in within "repetition" times the algorithm will be executed
        for trial in range(trials):
            self.best_value.best_obj_val = 1e-308
            repitionno_best, self.best_value.parameters = self.calculate_initial_parameterset(repetitions, initial_objs, initial_params)

            repetions_left = repetitions - repitionno_best

            # Main Loop of PA-DDS
            self.metric = self.calc_metric(metric)

            # important to set this field `generator_repetitions` so that
            # method `get_next_s_test` can generate exact parameters
            self.generator_repetitions = repetions_left

            for rep, x_curr, simulations in self.repeat(self.get_next_x_curr()):
                self.obj_func_current = self.postprocessing(rep, x_curr, simulations)
                num_imp = np.sum(self.obj_func_current <= self.best_value.best_obj_val)
                num_deg = np.sum(self.obj_func_current > self.best_value.best_obj_val)

                if num_imp == 0 and num_deg > 0:
                    self.dominance_flag = -1  # New solution is dominated by its parents
                else:  # Do dominance check only if new solution is not dominated by its parent
                    self.pareto_front, self.dominance_flag = nd_check(self.pareto_front, self.obj_func_current, x_curr.copy())
                    if self.dominance_flag != -1:  # means, that new parameter set is a new non-dominated solution
                        self.metric = self.calc_metric(metric)
                self.parameter_current = x_curr
                # update the new status structure
                self.status.params_max, self.status.params_min = self.parameter_current, self.parameter_current

            print('Best solution found has obj function value of ' + str(self.best_value.best_obj_val) + ' at '
                  + str(repitionno_best) + '\n\n')
            debug_results.append({"sbest": self.best_value.parameters , "objfunc_val": self.best_value.best_obj_val})

        self.final_call()
        return debug_results

    def calc_metric(self, metric):
        """
        calculate / returns metric field
        :return: set of metric of choice
        """
        if metric == "ones":
            return np.array([1] * self.pareto_front.shape[0])
        elif metric == "crowd_distance":
            return crowd_dist(np.array([w for w in self.pareto_front[:,0]]))
        elif metric == "chc":
            return chc(np.array([w for w in self.pareto_front[:,0]]))
        elif metric == "hvc":
            return self.hvc(np.array([w for w in self.pareto_front[:,0]]))
        else:
            raise AttributeError("metric argument is invalid")

    def calc_initial_pareto_front(self, its):
        """
        calculate the initial pareto front
        :param its: amount of initial parameters
        """


        dominance_flag = -1
        for i in range(its):
            for j in range(self.number_of_parameters):
                if self.best_value.parameters.as_int[j]:
                    self.parameter_current[j] = self.np_random.randint(self.best_value.parameters.minbound[j],
                                                                       self.best_value.parameters.maxbound[j])
                else:
                    self.parameter_current[j] = self.best_value.parameters.minbound[j] + self.parameter_range[
                        j] * self.np_random.rand()  # uniform random

            id, params, model_simulations = self.simulate((range(len(self.parameter_current)), self.parameter_current))
            self.obj_func_current = self.getfitness(simulation=model_simulations, params=self.parameter_current)
            # First value will be used to initialize the values
            if i == 0:
                self.pareto_front = np.vstack(
                    [self.pareto_front[0], np.array([self.obj_func_current.copy(), self.parameter_current.copy() + 0])])
            else:
                (self.pareto_front, dominance_flag) = nd_check(self.pareto_front, self.obj_func_current,
                                                               self.parameter_current.copy())

        self.dominance_flag = dominance_flag


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
            new_value = self.dds_generator.neigh_value_mixed(previous_x_curr, r, dec_var - 1,  self.min_bound[dec_var - 1],self.max_bound[dec_var - 1])

            new_x_curr[dec_var - 1] = new_value  # change relevant decision variable value in s_test

        return new_x_curr


def nd_check(nd_set_input, objective_values, parameter_set):
    """
    It is the Non Dominated Solution Check (ND Check)

    Meaning of dominance_flag
    dominance_flag = -1: parameter_set is dominated by pareto front
    dominance_flag =  0: parameter_set is a new non-dominated solution but not dominating
    dominance_flag =  1: parameter_set is a new non-dominated solution and dominating


    :param nd_set_input: Pareto Front
    :param objective_values: objective values
    :param parameter_set: parameter set
    :return: a new pareto front and a value if it was dominated or not (0,1,-1)
    """
    # Algorithm from PADDS Matlab Code

    nd_set = deepcopy(nd_set_input)
    dominance_flag = 0

    # These are simply reshaping problems if we want to loop over arrays but we have a single float given
    try:
        like_struct_len = objective_values.shape[0]
    except IndexError:
        objective_values = objective_values.reshape((1,))
        like_struct_len = objective_values.shape[0]
    try:
        # TODO delete pareto_high
        pareto_high = nd_set.shape[1]
    except IndexError:
        nd_set = nd_set.reshape(1, nd_set.shape[0])
        pareto_high = nd_set.shape[1]

    i = -1  # solution counter
    while i < nd_set.shape[0] - 1:
        i += 1

        try:
            _ = objective_values < nd_set[i][0]
        except ValueError:
            nd_set[i][0] = np.array([np.inf]*objective_values.shape[0])

        num_eql = np.sum(objective_values == nd_set[i][0])
        num_imp = np.sum(objective_values < nd_set[i][0])
        num_deg = np.sum(objective_values > nd_set[i][0])

        if num_imp == 0 and num_deg > 0:  # parameter_set is dominated
            dominance_flag = -1
            return nd_set, dominance_flag
        elif num_eql == like_struct_len:
            # Objective functions are the same for parameter_set and archived solution i
            # TODO check if this line still works
            nd_set[i][0], nd_set[i][1] = objective_values, parameter_set  # Replace solution i in ND_set with X
            dominance_flag = 0  # X is non - dominated
            return nd_set, dominance_flag
        elif num_imp > 0 and num_deg == 0:  # X dominates ith solution in the ND_set
            nd_set = np.delete(nd_set, i, 0)
            i = i - 1
            dominance_flag = 1

    if nd_set.size == 0:  # that means the array is completely empty
        nd_set = np.array([objective_values, parameter_set])  # Set solution i in ND_set with X
    else:  # If X dominated a portion of solutions in ND_set
        nd_set = np.vstack(
            [nd_set, np.array([objective_values, parameter_set])])  # Add the new solution to the end of ND_set (for later use and comparing!

    return nd_set, dominance_flag


def crowd_dist(points):
    """
    This function calculates the Normalized Crowding Distance for each member
    or "points". Deb book p236
     The structure of PF_set is as follows:
     PF_set = [obj_1, obj_2, ... obj_m, DV_1, DV_2, ..., DV_n]

     e.g. v = np.array([[1,10], [2,9.8], [3,5], [4,4], [8,2], [10,1]]); CDInd = crowd_dist(v)

    :param points: mainly is this a pareto front, but could be any set of data which a crowd distance should be calculated from
    :return: the crowd distance distance
    """

    # Normalize Objective Function Space
    try:  # Python / Numpy interprets arrays sometimes with wrong shape, this is a fix
        length_x = points.shape[1]
    except IndexError:
        points = points.reshape((1, points.shape[0]))
        length_x = points.shape[1]

    max_f = np.nanmax(points, 0)
    min_f = np.nanmin(points, 0)

    levels = (max_f == min_f)
    length_y = points.shape[0]

    indicies = np.array(range(length_x))[levels]
    max_f[indicies] += 1

    MAX_f = np.transpose(max_f.repeat(length_y).reshape((length_x, length_y)))
    MIN_f = np.transpose(min_f.repeat(length_y).reshape((length_x, length_y)))

    points = np.divide(points - MIN_f, MAX_f - MIN_f)

    # resave Length
    length_x = points.shape[1]
    length_y = points.shape[0]

    pointsWithNoNan = points[:, ~np.any(np.isnan(points), axis=0)]
    # Initialization
    zero_column = np.array([[0] * length_y]).reshape((length_y, 1))
    index_column = np.array(range(length_y)).reshape((length_y, 1))
    temp = np.concatenate((pointsWithNoNan, zero_column, index_column), 1)
    ij = temp.shape[1] - 2
    endpointIndx = np.array([0] * 2 * length_x)

    # Main Calculation
    if length_y <= length_x + 1:  # Less than or equal # obj + 1 solutions are non-dominated
        temp[:, ij] = 1  # The crowding distance is 1 for all archived solutions
        return temp[:, ij]
    else:  # More than 2 solutions are non - dominated
        for i in range(length_x):
            #  https://stackoverflow.com/a/22699957/5885054
            temp = temp[temp[:, i].argsort()]
            temp[0, ij] = temp[0, ij] + 2 * (temp[1, i] - temp[0, i])
            temp[length_y - 1, ij] = temp[length_y - 1, ij] + 2 * (temp[length_y - 1, i] - temp[length_y - 2, i])

            for j in range(1, length_y - 1):
                temp[j, ij] = temp[j, ij] + (temp[j + 1, i] - temp[j - 1, i])

            endpointIndx[2 * (i - 1) + 0] = temp[0, -1]
            endpointIndx[2 * (i - 1) + 1] = temp[-1, -1]

    #  Endpoints of Pareto Front
    temp = temp[temp[:, temp.shape[
                            1] - 1].argsort()]  # Sort points based on the last column to restore the original order of points in the archive
    endpointIndx = np.unique(endpointIndx)

    non_endpointIndx = np.array(range(length_y)).reshape((length_y, 1))
    non_endpointIndx = np.delete(non_endpointIndx, endpointIndx, 0)

    non_endpointIndx = non_endpointIndx.reshape((non_endpointIndx.shape[0]))

    Y = points[endpointIndx, :]
    X = points[non_endpointIndx, :]
    IDX = dsearchn(X, Y)  # Identify the closest point in the objective space to each endpoint (dsearchn in Matlab)
    if IDX.size > 0:
        for i in range(endpointIndx.shape[0]):
            temp[endpointIndx[i], ij] = np.nanmax([temp[endpointIndx[i], ij], temp[non_endpointIndx[IDX[
                i]], ij]])  # IF the closest point to the endpoint has a higher CD value, assign that to the endpoint; otherwise leave the CD value of the endpoint unchanged
    return temp[:, ij]


def dsearchn(x, y):
    """
    Implement Octave / Matlab dsearchn without triangulation
    :param x: Search Points in
    :param y: Were points are stored
    :return: indices of points of x which have minimal distance to points of y
    """
    IDX = []
    for line in range(y.shape[0]):
        distances = np.sqrt(np.nansum(np.power(x - y[line, :], 2), axis=1))
        found_min_dist_ind = (np.nanmin(distances, axis=0) == distances)
        length = found_min_dist_ind.shape[0]
        IDX.append(np.array(range(length))[found_min_dist_ind][0])
    return np.array(IDX)


class HVC():
    def __init__(self, *args, **kwargs):
        self.fakerandom =  ('fakerandom' in kwargs and kwargs['fakerandom']) or ('np_random' in kwargs)
        self.has_random_class = ('np_random' in kwargs)
        self.random_class = (self.has_random_class and kwargs['np_random'])
        from deap.tools._hypervolume import hv
        self.hv = hv.hypervolume
        self.h_vol_c = self.hv_wrapper

    def hv_wrapper(self, points):
        ref = np.max(points, axis=0)
        return self.hv(points, ref)

    def _set_np_random(self,f_rand):
        self.random_class = f_rand
        self.has_random_class = True

    def hype_indicator_sampled(self,points, bounds, nrOfSamples):
        try:
            nrP, dim = points.shape
        except ValueError:
            nrP, dim = points.shape[0], 1

        F = np.array([0] * nrP)
        BoxL = np.min(points, 0)

        S = np.dot(self.__rand(nrOfSamples, dim), np.diag(bounds - BoxL)) + np.dot(np.ones([nrOfSamples, dim]), np.diag(BoxL))

        dominated = np.array([0] * nrOfSamples)
        dominated_ind = np.array([0] * nrOfSamples)

        ROWS = np.transpose(np.array([range(nrOfSamples)]))

        for j in range(nrP):
            B = S - np.repeat([points[j, :]], S.shape[0], 0)
            ind = np.sum(B >= 0, 1) == dim
            dominated[ind] += 1
            dominated_ind[ind] = j
            Index = np.where(dominated == 2)
            S = np.delete(S, Index, 0)
            ROWS = np.delete(ROWS, Index, 0)
            dominated = np.delete(dominated, Index, 0)
            dominated_ind = np.delete(dominated_ind, Index, 0)

        Index = np.array(range(S.shape[0]))
        Logical = dominated == 1
        ind = np.transpose(Index[Logical])
        Index = dominated_ind[ind]

        Index = np.sort(Index)

        for j in Index:
            F[j] = F[j] + 1
        F = np.transpose(F) * np.prod(bounds - BoxL) / nrOfSamples
        return F  # transpose??


    def hv_apprx(self,points):
        p_xlen = np.shape(points)[0]
        p_ylen = np.shape(points)[1]
        indexis = np.array(range(p_xlen)) + 1
        temp = np.hstack((points, indexis.reshape(p_xlen, 1)))
        endpointIndx = np.zeros(2 * p_ylen)

        for i in range(p_ylen):
            temp = self.sortrows(temp, i)

            endpointIndx[2 * (i)] = temp[0, -1]
            endpointIndx[2 * (i) + 1] = temp[-1:, -1:]
        endpointIndx = np.int32(np.unique(endpointIndx) - 1)

        nrOfSamples = np.max([10000, 2 * p_xlen])  # Dictates the accuracy of approximation

        HVInf = self.hype_indicator_sampled(points, np.array([1] * p_ylen), nrOfSamples)

        nonZero_Indx = np.where(HVInf > 0)[0]
        Y = points[endpointIndx, :]
        X = points[nonZero_Indx, :]

        IDX = dsearchn(X, Y)

        for i in range(len(endpointIndx)):
            if len(IDX) > 0:
                HVInf[endpointIndx[i]] = HVInf[nonZero_Indx[IDX[i]]]
            else:
                HVInf[endpointIndx[i]] = 1

        return HVInf


    def __rand(self, x, y):
        if self.fakerandom:
            if self.has_random_class:
                return self.random_class.rand(x,y)
            else:
                dim = x * y + 1
                step = 1.0 / dim
                data = np.arange(step, 1, step)[0:x * y]
                if y <= 1:
                    reshaper = [x]
                else:
                    reshaper = [x, y]
                return data.reshape(reshaper)
        else:
            return np.random.rand(x,y)



    def sortrows(self, arr, index):
        """
        https://gist.github.com/stevenvo/e3dad127598842459b68
        :param arr:
        :param index:
        :return:
        """
        return arr[arr[:, index].argsort()]

    def hv_exact(self,points):
        p_xlen = np.shape(points)[0]
        p_ylen = np.shape(points)[1]

        indexis = np.array(range(p_xlen)) + 1

        temp = np.hstack((points, indexis.reshape(p_xlen, 1)))
        endpointIndx = np.zeros(2 * p_ylen)

        for i in range(p_ylen):
            temp = self.sortrows(temp, i)

            endpointIndx[2 * (i)] = temp[0, -1]
            endpointIndx[2 * (i) + 1] = temp[-1:, -1:]

        endpointIndx = np.int32(np.unique(endpointIndx) - 1)

        if len(endpointIndx) == p_xlen:
            return np.ones(p_xlen)

        indexis = np.delete(indexis, endpointIndx)

        totalHV = self.h_vol_c(points)

        HVInf = np.zeros(p_xlen)

        for i in range(len(indexis)):
            y = points
            y = np.delete(y, indexis[i] - 1, 0)
            subhv = self.h_vol_c(y)

            HVInf[indexis[i] - 1] = totalHV - subhv

        non_endpointIndx = np.array(range(p_xlen))
        non_endpointIndx = non_endpointIndx[[i for i in range(p_xlen) if i not in endpointIndx]]

        Y = points[endpointIndx, :]
        X = points[non_endpointIndx, :]
        IDX = dsearchn(X, Y)

        for i in range(len(endpointIndx)):
            if len(IDX) > 0:
                HVInf[endpointIndx[i]] = HVInf[non_endpointIndx[IDX[i]]]
            else:
                HVInf[endpointIndx[i]] = 1

        return HVInf


    def __call__(self, points):
        p_xlen = np.shape(points)[0]
        p_ylen = np.shape(points)[1]

        if p_xlen <= p_ylen + 1:
            return np.array([1] * p_xlen)

        max_f = np.max(points, 0)
        min_f = np.min(points, 0)
        max_f = max_f.reshape(1, p_ylen)
        min_f = min_f.reshape(1, p_ylen)

        ind = max_f == min_f
        max_f[ind] = max_f[ind] + 1

        MAX_f = np.repeat(max_f, p_xlen, 0)
        MIN_f = np.repeat(min_f, p_xlen, 0)

        points = (points - MIN_f) / (MAX_f - MIN_f)

        if p_xlen <= p_ylen + 1:
            return np.array([1] * p_xlen)

        if p_ylen <= 4:
            return self.hv_exact(points)
        elif p_ylen > 4:
            return self.hv_apprx(points)



def chc(points):
    """
    function CHC_metric = CHC(points)
    Comments
    Original written by Masoud Asadzadeh, University of Waterloo, June 03 2011

    This code calls the qhull by SciPy (see also http://www.qhull.org/)

    This function calculates Convex Hull Contribution CHC. See:
    Asadzadeh, M., B. A. Tolson, and D. H. Burn (2014), A new selection metric for multiobjective hydrologic model calibration, doi:10.1002/2013WR014970.
    There are four mutually exclusive sets of points in CH:
    i. Points inside the convex hull
    ii. Vertices of top facet only
    iii. Vertices of bottom facets only
    iv. Vertices in the intersection of top and bottom facets

    CHC is ZERO for points in i and ii. CHC for points iii is calculated as
    their contribution to the volume of the convex hull of set of "points".
    CHC for points iv is calculated as the CHC of closest point in iii.

    e.g. CHC_metric = CHC([1 10; 2 9.8; 3 5; 4 4; 8 2; 10 1])
    Normalize the objective space
    :param points:
    :return:
    """

    max_f = np.max(points, 0)
    min_f = np.min(points, 0)
    p_xlen = np.shape(points)[0]
    p_ylen = np.shape(points)[1]

    max_f = max_f.reshape(1, p_ylen)
    min_f = min_f.reshape(1, p_ylen)
    MAX_f = np.repeat(max_f, p_xlen, 0)
    MIN_f = np.repeat(min_f, p_xlen, 0)

    CHC_metric = np.array([0.0] * p_xlen)

    points = (points - MIN_f) / (MAX_f - MIN_f)

    if p_xlen <= p_ylen + 1:
        return np.array([1] * p_xlen)

    try:
        hull = ConvexHull(points)
    except ValueError:
        hull = None
        return np.array([1.] * np.max(points.shape))

    Totalv = hull.volume
    # hull.vertices
    norm = hull.equations[:, :-1]
    num = hull.nsimplex
    vertices = []

    for s in hull.simplices:
        vertices.append(list(s + 1))
    vertices = np.array(vertices)

    all_CHpts_ind = np.unique(vertices)
    ZEROind = all_CHpts_ind == 0
    all_CHpts_ind[ZEROind] = []

    # Identify points in groups ii, iii, iv as defined above
    top_facets_ind = np.min(norm, 1) >= 0  # facets with outward norm that has only non-negative components

    ii_iv_CHpts_ind = np.unique(vertices[top_facets_ind])  # points on top of CH, i.e. groups ii and iv

    ZEROind = ii_iv_CHpts_ind == 0
    ii_iv_CHpts_ind[ZEROind] = []
    other_facets_ind = np.array(range(norm.shape[0]))  # All facets
    other_facets_ind = np.delete(other_facets_ind, other_facets_ind[top_facets_ind])

    iii_iv_CHpts_ind = np.unique(vertices[other_facets_ind, :])  # points on bottom of CH, i.e. groups iii and iv

    ZEROind = iii_iv_CHpts_ind == 0

    iii_iv_CHpts_ind = np.delete(iii_iv_CHpts_ind, iii_iv_CHpts_ind[ZEROind])

    bor_ind = np.array([y in ii_iv_CHpts_ind for y in iii_iv_CHpts_ind])

    bor_CHpts_ind = iii_iv_CHpts_ind[bor_ind]
    bot_CHpts_ind = iii_iv_CHpts_ind
    bot_CHpts_ind = bot_CHpts_ind[bor_ind == False]  # Remove border points from bottom points

    # When number of bottom points and border points are not enough to form CH
    if bot_CHpts_ind.shape[0] == 0:
        CHC_metric[bot_CHpts_ind - 1] = 1
        CHC_metric[bor_CHpts_ind - 1] = 1
        return CHC_metric

    for i in range(bot_CHpts_ind.shape[0]):
        y = points[all_CHpts_ind - 1]  # Only consider points that are on the vertices of convex hull
        # Meaning that forget the points inside the convex hull
        ind = np.array([j == bot_CHpts_ind[i] for j in all_CHpts_ind])
        y = y[ind == False]
        try:
            convhull = ConvexHull(y)
            Sub_v = convhull.volume

            if Sub_v > Totalv:  # just in case of numerical issues
                CHC_metric[bot_CHpts_ind[i] - 1] = 0
            else:
                CHC_metric[bot_CHpts_ind[i] - 1] = Totalv - Sub_v
        except QhullError:
            # an error occured
            CHC_metric = np.array([0.0] * p_xlen)
            CHC_metric[bot_CHpts_ind - 1] = 1
            CHC_metric[bor_CHpts_ind - 1] = 1
            return CHC_metric

    if np.max(CHC_metric) == 0:  # In case no solution has valid CHC value
        CHC_metric[bot_CHpts_ind - 1] = 1
        CHC_metric[bor_CHpts_ind - 1] = 1

    Y = points[bor_CHpts_ind - 1, :]
    X = points[bot_CHpts_ind - 1, :]
    IDX = dsearchn(X, Y)
    for i in range(bor_CHpts_ind.shape[0]):
        CHC_metric[bor_CHpts_ind[i] - 1] = CHC_metric[bot_CHpts_ind[IDX[i]] - 1]

    return CHC_metric
