import numpy as np
from copy import deepcopy

from spotpy.algorithms.dds import DDSGenerator
from . import _algorithm
from spotpy.parameter import ParameterSet
import copy


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

        self.status.params = ParameterSet(self.parameter())

        self.dds_generator = DDSGenerator(self.np_random)
        # self.generator_repetitions will be set in `sample` and is needed to generate a
        # generator which sends back actual parameter s_test
        self.generator_repetitions = -1
        self.pareto_front = np.array([])
        self.dominance_flag = -2
        self.obj_func_current = None
        self.parameter_current = None

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
            if self.dominance_flag == -1:  # if the last generated solution was dominated
                index = self.roulette_wheel(self.metric)
                self.status.params = self.pareto_front[index, self.like_struct_len:]
                self.status.objectivefunction = self.pareto_front[index,:self.like_struct_len]
            else: # otherwise use the last generated solution
                self.status.objectivefunction = self.obj_func_current
                self.status.params = self.parameter_current

            # This line is needed to get an array of data converted into a parameter object
            self.fix_status_params_format()

            yield rep, self.calculate_next_s_test(self.status.params, rep, self.generator_repetitions, self.r)

    def calculate_initial_parameterset(self, repetitions, x_initial):
        self.obj_func_current = np.array([0.0] * self.like_struct_len)
        self.parameter_current = np.array([0.0] * self.number_of_parameters)
        self.parameter_range = self.status.params.maxbound - self.status.params.minbound
        self.pareto_front = np.array([np.append([np.inf] * self.like_struct_len, [0] * self.number_of_parameters)])

        if len(x_initial) == 0:
            initial_iterations = np.int(np.max([5, round(0.005 * repetitions)]))
            self.calc_initial_pareto_front(initial_iterations)
        elif x_initial.shape[1] != self.number_of_parameters:
            raise ValueError("User specified 'x_initial' has not the same length as available parameters")
        else:
            if not (np.all(x_initial <= self.status.params.maxbound) and np.all(
                    x_initial >= self.status.params.minbound)):
                raise ValueError("User specified 'x_initial' but the values are not within the parameter range")
            initial_iterations = x_initial.shape[0]

            for i in range(x_initial.shape[0]):
                if x_initial.shape[1] == self.like_struct_len + self.number_of_parameters:
                    self.obj_func_current = x_initial[i, :self.like_struct_len]
                    self.parameter_current = x_initial[i, self.like_struct_len:]
                else:
                    self.parameter_current = x_initial[i]
                    self.obj_func_current = self.getfitness(simulation=[], params=self.parameter_current)

                if i == 0:  # Initial value
                    self.pareto_front = np.array([np.append(self.obj_func_current, self.parameter_current)])
                    dominance_flag = 1
                else:
                    self.pareto_front, dominance_flag = nd_check(self.pareto_front, self.obj_func_current, self.parameter_current)
                self.dominance_flag = dominance_flag

        return initial_iterations, copy.deepcopy(self.parameter_current)

    def sample(self, repetitions, trials=1, x_initial=np.array([]), metric="ones"):
        # every iteration a map of all relevant values is stored, only for debug purpose.
        # Spotpy will not need this values.
        debug_results = []
        print('Starting the PADDS algotrithm with ' + str(repetitions) + ' repetitions...')
        self.set_repetiton(repetitions)
        self.number_of_parameters = len(self.status.params) # number_of_parameters is the amount of parameters

        # Users can define trial runs in within "repetition" times the algorithm will be executed
        for trial in range(trials):
            self.status.objectivefunction = 1e-308
            repitionno_best, self.status.params = self.calculate_initial_parameterset(repetitions, x_initial)

            repetions_left =  repetitions - repitionno_best

            # Main Loop of PA-DDS
            self.metric = self.calc_metric(metric)

            # important to set this field `generator_repetitions` so that
            # method `get_next_s_test` can generate exact parameters
            self.generator_repetitions = repetions_left

            for rep, x_curr, simulations in self.repeat(self.get_next_x_curr()):
                self.obj_func_current = self.postprocessing(rep, x_curr, simulations)
                num_imp = np.sum(self.obj_func_current <= self.status.objectivefunction)
                num_deg = np.sum(self.obj_func_current > self.status.objectivefunction)

                if num_imp == 0 and num_deg > 0:
                    self.dominance_flag = -1 # New solution is dominated by its parents
                else: # Do dominance check only if new solution is not dominated by its parent
                    self.pareto_front, self.dominance_flag = nd_check(self.pareto_front, self.obj_func_current, x_curr)
                    if self.dominance_flag != -1: # means, that new parameter set is a new non-dominated solution
                        self.metric = self.calc_metric(metric)
                self.parameter_current = x_curr

            print('Best solution found has obj function value of ' + str(self.status.objectivefunction) + ' at '
                  + str(repitionno_best) + '\n\n')
            debug_results.append({"sbest": self.status.params, "objfunc_val":self.status.objectivefunction})

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
            return crowd_dist(self.pareto_front[:,0:self.like_struct_len])
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
                if self.status.params.as_int[j]:
                    self.parameter_current[j] = self.np_random.randint(self.status.params.minbound[j], self.status.params.maxbound[j])
                else:
                    self.parameter_current[j] = self.status.params.minbound[j] + self.parameter_range[j] * self.np_random.rand()  # uniform random

            id, params, model_simulations = self.simulate((range(len(self.parameter_current)), self.parameter_current))
            self.obj_func_current = self.getfitness(simulation=model_simulations, params=self.parameter_current)
            # First value will be used to initialize the values
            if i == 0:
                self.pareto_front = np.vstack([self.pareto_front[0], np.append(self.obj_func_current, self.parameter_current)])
            else:
                (self.pareto_front, dominance_flag) = nd_check(self.pareto_front, self.obj_func_current, self.parameter_current)

        self.dominance_flag = dominance_flag

    def fix_status_params_format(self):
        start_params = ParameterSet(self.parameter())
        start_params.set_by_array([j for j in self.status.params])
        self.status.params = start_params


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
                new_value = self.dds_generator.neigh_value_mixed(previous_x_curr, r, j)
                new_x_curr[j] = new_value  # change relevant dec var value in x_curr

        if dvn_count == 0:  # no DVs selected at random, so select ONE
            dec_var = np.int(np.ceil(amount_params * self.np_random.rand()))
            new_value = self.dds_generator.neigh_value_mixed(previous_x_curr, r, dec_var - 1)

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
        pareto_high = nd_set.shape[1]
    except IndexError:
        nd_set = nd_set.reshape(1,nd_set.shape[0])
        pareto_high = nd_set.shape[1]

    i = -1  # solution counter
    while i < nd_set.shape[0]-1:
        i += 1
        num_eql = np.sum(objective_values == nd_set[i, :like_struct_len])
        num_imp = np.sum(objective_values < nd_set[i, :like_struct_len])
        num_deg = np.sum(objective_values > nd_set[i, :like_struct_len])

        if num_imp == 0 and num_deg > 0:  # parameter_set is dominated
            dominance_flag = -1
            return (nd_set, dominance_flag)
        elif num_eql == like_struct_len:
            # Objective functions are the same for parameter_set and archived solution i
            # TODO check if this line still works
            nd_set[i] = np.append(objective_values, parameter_set)  # Replace solution i in ND_set with X
            dominance_flag = 0  # X is non - dominated
            return nd_set, dominance_flag
        elif num_imp > 0 and num_deg == 0:  # X dominates ith solution in the ND_set
            nd_set = np.delete(nd_set, i, 0)
            i = i - 1
            dominance_flag = 1

    if nd_set.size == 0:  # that means the array is completely empty
        nd_set = np.array([np.append(objective_values, parameter_set)])  # Set solution i in ND_set with X
    else:  # If X dominated a portion of solutions in ND_set
        nd_set = np.vstack(
            [nd_set, np.append(objective_values, parameter_set)])  # Add the new solution to the end of ND_set (for later use and comparing!

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
    try: # Python / Numpy interprets arrays sometimes with wrong shape, this is a fix
        length_x = points.shape[1]
    except IndexError:
        points = points.reshape((1, points.shape[0]))
        length_x = points.shape[1]


    max_f = np.max(points,0)
    min_f = np.min(points,0)
    levels = (max_f == min_f)
    length_y = points.shape[0]


    indicies = np.array(range(length_x))[levels]
    max_f[indicies] += 1


    MAX_f = np.transpose(max_f.repeat(length_y).reshape((length_x,length_y)))
    MIN_f = np.transpose(min_f.repeat(length_y).reshape((length_x,length_y)))


    points = np.divide(points - MIN_f,MAX_f - MIN_f)

    # resave Length
    length_x = points.shape[1]
    length_y = points.shape[0]

    # Initialization
    zero_column = np.array([[0] * length_y]).reshape((length_y, 1))
    index_column = np.array(range(length_y)).reshape((length_y,1))
    temp = np.concatenate((points, zero_column, index_column), 1)
    ij = temp.shape[1] - 2
    endpointIndx = np.array([0]*2*length_x)

    # Main Calculation
    if length_y <= length_x + 1:  # Less than or equal # obj + 1 solutions are non-dominated
        temp[:, ij] = 1   # The crowding distance is 1 for all archived solutions
        return temp[:, ij]
    else: # More than 2 solutions are non - dominated
        for i in range(length_x):
            #  https://stackoverflow.com/a/22699957/5885054
            temp = temp[temp[:,i].argsort()]
            temp[0, ij] = temp[0, ij] + 2 * (temp[1, i] - temp[0, i])
            temp[length_y-1, ij] = temp[length_y-1,ij] + 2*(temp[length_y-1,i] - temp[length_y-2,i])

            for j in range(1, length_y-1):
                temp[j, ij] = temp[j, ij] + (temp[j + 1, i] - temp[j - 1, i])

            endpointIndx[2 * (i - 1) + 0] = temp[0, -1]
            endpointIndx[2 * (i - 1) + 1] = temp[-1, -1]

    #  Endpoints of Pareto Front
    temp = temp[temp[:,temp.shape[1]-1].argsort()]   # Sort points based on the last column to restore the original order of points in the archive
    endpointIndx = np.unique(endpointIndx)


    non_endpointIndx = np.array(range(length_y)).reshape((length_y,1))
    non_endpointIndx=np.delete(non_endpointIndx, endpointIndx, 0)

    non_endpointIndx = non_endpointIndx.reshape((non_endpointIndx.shape[0]))

    Y = points[endpointIndx, :]
    X = points[non_endpointIndx, :]
    IDX = dsearchn(X,Y)   # Identify the closest point in the objective space to each endpoint (dsearchn in Matlab)
    if IDX.size > 0:
        for i in range(endpointIndx.shape[0]):
            temp[endpointIndx[i], ij] = np.max([temp[endpointIndx[i], ij],temp[non_endpointIndx[IDX[i]], ij]])   # IF the closest point to the endpoint has a higher CD value, assign that to the endpoint; otherwise leave the CD value of the endpoint unchanged

    return temp[:, ij]

def dsearchn(x,y):
    """
    Implement Octave / Matlab dsearchn without triangulation
    :param x: Search Points in
    :param y: Were points are stored
    :return: indices of points of x which have minimal distance to points of y
    """
    IDX = []
    for line in range(y.shape[0]):
        distances = np.sqrt(np.nansum(np.power(x - y[line, :], 2), axis=1))
        found_min_dist_ind = (np.min(distances, axis=0) == distances)
        length = found_min_dist_ind.shape[0]
        IDX.append(np.array(range(length))[found_min_dist_ind][0])
    return np.array(IDX)



