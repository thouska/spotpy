# -*- coding: utf-8 -*-
"""
EFAST algorithm after Reusser et al. transfered from R
Reusser, Dominik E., Wouter Buytaert, and Erwin Zehe. "Temporal dynamics of model parameter sensitivity for computationally expensive models with FAST (Fourier Amplitude Sensitivity Test)." Water Resources Research 47 (2011): W07551.

Further References:
CUKIER, R. I.; LEVINE, H. B. & SHULER, K. E. Non-Linear Sensitivity Analysis Of Multi-Parameter Model Systems Journal Of Computational Physics, 1978 , 26 , 1-42
CUKIER, R. I.; FORTUIN, C. M.; SHULER, K. E.; PETSCHEK, A. G. & SCHAIBLY, J. H. Study Of Sensitivity Of Coupled Reaction Systems To Uncertainties In Rate Coefficients .1. Theory Journal Of Chemical Physics, 1973 , 59 , 3873-3878
SCHAIBLY, J. H. & SHULER, K. E. Study Of Sensitivity Of Coupled Reaction Systems To Uncertainties In Rate Coefficients .2. Applications Journal Of Chemical Physics, 1973 , 59 , 3879-3888
CUKIER, R. I.; SCHAIBLY, J. H. & SHULER, K. E. Study Of Sensitivity Of Coupled Reaction Systems To Uncertainties In Rate Coefficients .3. Analysis Of Approximations Journal Of Chemical Physics, 1975 , 63 , 1140-1149

Transferred form R and implemented by Anna Herzog (2024)

Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Anna Herzog, Tobias Houska
"""

import numpy as np
import warnings

from . import _algorithm
from ..analyser import   efast_sensitivity, get_modelruns

class efast(_algorithm):
    """
    Efast Algorithm for (distributed) parameter Sensitivity after FAST algorithm according to Cukier 1975 or McRae 1982
    """

    _unaccepted_parameter_types = ()

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

        

        kwargs["algorithm_name"] = "EFast Sampler"
        super(efast, self).__init__(*args, **kwargs)

    # hard coded repetition and frequency values from R package Fast
    d_m_cukier75 = [4,8,6,10,20,22,32,40,38,26,56,62,46,76,96,60,86,126,134,112,92,128,154,196,34,416,106,208,328,198,382,88,348,186,140,170,284,568,302,438,410,248,448,388,596,217,100,488,166]
    min_runs_cukier75 = [np.nan, np.nan, 19, 39, 71, 91, 167, 243, 315, 403, 487, 579, 687, 907, 1019, 1223, 1367, 1655, 1919, 2087, 2351, 2771, 3087, 3427, 3555, 4091, 4467, 4795, 5679, 5763, 6507, 7103, 7523, 8351, 9187, 9667, 10211, 10775, 11339, 7467, 12891, 13739, 14743, 15743, 16975, 18275, 18927, 19907, 20759, 21803]
    omega_m_cukier75 = [np.nan, np.nan, 1, 5, 11, 1, 17, 23, 19, 25, 41, 31, 23, 87, 67, 73, 58, 143, 149, 99, 119, 237, 267, 283, 151, 385, 157, 215, 449, 163, 337, 253, 375, 441, 673, 773, 875, 873, 587, 849, 623, 637, 891, 943, 1171, 1225, 1335, 1725, 1663, 2019]
    d_m_mcrae82 = [4, 8, 6, 10, 20, 22, 32, 40, 38, 26, 56, 62, 46, 76, 96]
    min_runs_mcrae82 = [0, 15, 27, 47, 79, 99, 175, 251, 323, 411, 495, 587, 695, 915, 1027]
    omega_m_mcrae82 = [0, 3, 1, 5, 11, 1, 17, 23, 19, 25, 41, 31, 23, 87, 67]

    def freq_cukier(self, m, i = 1, omega_before = -1):
    
        """
        This function creates an array of independant frequencies accroding to 
        Cukier 1975 for usage in the fast method.

        Parameters
        ----------
        m: int
            number of parameters (frequencies) needed
        i: (intern) int
            internally used recursion counter
        omega_before: (intern) int
            internally used previous frequency

        Returns
        ----------
        value: np.array of float
            A 1d-Array of independant frequencies to the order of 4

        """

        if i <= 1:
            if m >= len(self.omega_m_cukier75):
                raise Exception("Parameter number m to high, not implemented") 
            else:
                o = self.omega_m_cukier75[m-1]
                return np.append(o, self.freq_cukier(m, i+1, o))
        else:
            o = omega_before + self.d_m_cukier75[m-i]
            if i == m:
                return o
            else:
                return np.append(o, self.freq_cukier(m, i+1, o))
        

    def freq_mcrae82(self, m, i = 1, omega_before = -1):
    
        """
        This function creates an array of independant frequencies accroding to 
        McRae 1982 for usage in the fast method.

        Parameters
        ----------
        m: int
            number of parameters (frequencies) needed
        i: (intern) int
            internally used recursion counter
        omega_before: (intern) int
            internally used previous frequency

        Returns
        ----------
        value: np.array of float
            A 1d-Array of independant frequencies to the order of 4

        """

        if i <= 1:
            if m >= len(self.omega_m_mcrae82):
                raise Exception("Parameter number m to high, not implemented") 
            else:
                o = self.omega_m_mcrae82[m-1]
                return np.append(o, self.freq_mcrae82(m, i+1, o))
        else:
            o = omega_before + self.d_m_mcrae82[m-i]
            if i == m:
                return o
            else:
                return np.append(o, self.freq_mcrae82(m, i+1, o))
        

    def s(self, m, factor = 1, cukier = True):

        """
        Function that generates a number of equally spaced values between -p1/2 
        and pi/2. The number is determined by the number of runs required for the
        FAST method for a number of parameters (min_runs_cukier or min_runs_mcrae)

        Parameters
        ----------
        m: int
            number of parameters/ frequencies
        factor: (optional) int
            used if more than the minimum required shall be generated
            default: the length of the returned array is the minimum number 
            required for the FAST time factor
        cukier: (optional) bool
            indicates weather to use the frequencies after Cukier or McRae
            Default is Cukier

        Returns
        ----------
        value: array of float
            an array of equally spaced values between -pi/2 and pi/2

        """

        if cukier:
            min_runs = self.min_runs_cukier75
        else:
            min_runs = self.min_runs_mcrae82
        
        r = np.round(min_runs[m-1]*factor)
        r_range = np.array(range(1,r+1))
        s = np.pi/r * (2*r_range-r-1)/2
    
        return s


    def S(self, m, factor = 1, cukier = True): # , par_names = np.nan, reorder = range(0,m)
    
        """
        Function to generate an array of values with provide the base for parameters 
        for the FAST method. It is usally not used directly but called from the 
        fast_parameters function.

        Parameters
        ----------
        m: int
            number of parameters/frequencies
        factor: (optional) int
            used to create more values than the minimum required.
        cukier: (optional) bool
            indicates weather to use the frequencies after Cukier or McRae
            Default is Cukier

        Returns
        ----------
        value: array of float
            an array with the shape (number of runs, parameters)
        """

        if cukier:
            omega = self.freq_cukier(m)
        else:
            omega = self.freq_mcrae82(m)
    
        tab = np.outer(self.s(m, factor, cukier), omega)
    
        toreturn = np.arcsin(np.sin(tab))/np.pi
    
        # naming array dimensions is not possible with numpy but convention would be toreturn.shape () = (runs, parameternames)
    
        return toreturn


    def rerange(self, data, min_goal = 0, max_goal = 1, center = np.nan):

        """
        This function performes a linear transformation of the data, such that
        afterwards range(data) = (theMin, theMax)

        Parameters
        ----------
        data: array of flaot
            an array with the data to transform
            in this case the parameter distribution generated by function S
        min_goal: float
            the new minimum value (lower parameter bound)
        max_goal: float
            the new maximum value (upper parameter bound)
        center: (optional) float
            indicates which old value should become the new center
            default: (max_goal+min_goal)/2

        Returns
        ----------
        value: array of float
            array with the transformed data
    
        """
    
        min_data = min(data)
        max_data = max(data)
    
        if np.isnan(center):
            max_data = max_data
            dat = data - min_data
            dat = dat/(max_data-min_data) * (max_goal-min_goal)
            dat = dat + min_goal
            return(dat)
        else:
            # split linear transformation, the data is split into two segments, one blow center, one above, 
            # each segment undergoes it's own linear transformation
            below = data <= center
            above = data > center
            new_data = np.copy(data)
            np.place(new_data, below, self.rerange(np.insert(data[below], 0, center), min_goal = min_goal, max_goal= max_goal/2)[1:])
            np.place(new_data, above, self.rerange(np.insert(data[above], 0, center), min_goal = max_goal/2, max_goal= max_goal)[1:])
            return(new_data)
    

    def fast_parameters(self, minimum, maximum, cukier = True, factor = 1, logscale = np.nan, names = np.nan):
    
        """
        Function for the creation of a FAST Parameter set based on thier range

        Parameters
        ----------
        minimum: array of float 
            array containing the lower parameter boundries
        maximum: array of float 
            array containing the upper parameter boundries
        names: (optional) str
            array containing the parameter names
        factor: (optional) int
            used to create more values than the minimum required.
        cukier: (optional) bool
            indicates weather to use the frequencies after Cukier or McRae
            Default is Cukier
        logscale: (optional) bool
            array containing bool values indicating weather a parameter is varied
            on a logarithmic scale. In that case minimum and maximum are exponents

        Returns
        ----------
        value: array of float
            an array with the shape (number of runs, parameters) containing the 
            required parameter sets

        """
        
        if np.isnan(logscale):
            logscale = np.full(minimum.shape, False)
        if np.isnan(names):
            names = ["P"+str(i) for i in range(minimum.shape[0])]
            names = np.array(names)  
    
        n = len(minimum)
    
        if (n != len(maximum)):
            raise Exception("Expecting minimum and maximum of same size") 
        elif(n != len(names)):
            raise Exception("Expecting minimum and names of same size") 
        elif(n != len(logscale)):
            raise Exception("Expecting minimum and logscale of same size") 
    
        toreturn = self.S(m=n, cukier = cukier, factor = factor) #par_names = names, 
    
        for i in range(0,n):
            toreturn[:,i] = self.rerange(toreturn[:,i], minimum[i], maximum[i])
            if logscale[i]:
                toreturn[:,i] = 10**toreturn[:,i]
            
        return toreturn


    def sample(self, repetitions, cukier = True, factor = 1, logscale = np.nan):
        """
        Samples from the EFAST algorithm.

        Input
        ----------
        repetitions: int
            Maximum number of runs.
        """

        print("generating eFAST Parameters")
        # Get the names of the parameters to analyse
        names = self.parameter()["name"]
        # Get the minimum and maximum value for each parameter from the
        # distribution
        parmin, parmax = self.parameter()["minbound"], self.parameter()["maxbound"]
        self.numberf = len(parmin)
        min_runs = self.min_runs_cukier75[self.numberf-1]

        if min_runs > repetitions:
            warnings.warn("specified number of repetitions is smaller than minimum required number for FAST analysis!\n"+
                          "Simultions will be perfomed but eFAST analysis might not be possible")
        else:
            repetitions = min_runs
            print("Number of specified repetitions equals or exeeds number of minimum required repetitions for eFAST analysis\n"+
                  "programm will stop after required runs.")

        self.repetitions = repetitions
        self.set_repetiton(repetitions)

        # Generate Matrix with eFAST parameter sets
        N = self.fast_parameters(parmin, parmax, cukier, factor, logscale)

        print("Starting the eFast algorithm with {} repetitions...".format(repetitions))

        # A generator that produces parametersets if called
        param_generator = (
            (rep, N[rep,:]) for rep in range(int(repetitions))
        )
        for rep, randompar, simulations in self.repeat(param_generator):
            # A function that calculates the fitness of the run and the manages the database
            self.postprocessing(rep, randompar, simulations)

            if self.breakpoint == "write" or self.breakpoint == "readandwrite":
                if rep >= lastbackup + self.backup_every_rep:
                    work = (rep, N[rep,:])
                    self.write_breakdata(self.dbname, work)
                    lastbackup = rep
        self.final_call()

    def calc_sensitivity(self, results, dbname, cukier = True):
        """
        Function to calcultae the sensitivity for a data series (eg. a time series)

        Parameters
        ----------

        data: np.array of float
            spopty restults array
            data array containing the model output used for the sensitivity calculation 
            with one row per parameter set
    
        numberf : int
            number of parameters used

        xval: int, optional
    
        """ 
        print("call analyser")
        data = get_modelruns(results)
        
        # convert array of void to array of float
        mod_results = np.full((data.shape[0], len(data[0])), np.nan)

        for index in range(len(data)):        
            mod_results[index, :] = list(data[index])

        numberf = len(self.parameter()["minbound"])

        # sens_data = np.full((len(results[0]), numberf), np.nan)

        if cukier:
            t_runs = self.min_runs_cukier75[numberf-1]
            t_freq = self.freq_cukier(numberf)
        else: 
            t_runs = self.min_runs_mcrae82[numberf-1]
            t_freq = self.freq_mcrae82(numberf)
    
        # Get the names of the parameters to analyse
        names = ["par" + name for name in self.parameter()["name"]]

        f = open(dbname+".csv", 'w+')
        np.savetxt(f, [names], "%s", delimiter=",")
        
        for index in range(mod_results.shape[1]):    
            x_data = mod_results[:, index]
            sens_data = efast_sensitivity(x_data, numberf, t_runs, t_freq)
            np.savetxt(f, [sens_data], delimiter=",", fmt='%1.5f')
        
        f.close()