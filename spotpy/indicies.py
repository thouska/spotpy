# -*- coding: utf-8 -*-
'''
Copyright (c) 2017 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Benjamin Manns
:paper: Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: 
SPOTting Model Parameters Using a Ready-Made Python Package, 
PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015.
This package enables the comprehensive use of different Bayesian and Heuristic calibration 
techniques in one Framework. It comes along with an algorithms folder for the 
sampling and an analyser class for the plotting of results by the sampling.
:dependencies: - Numpy >1.8 (http://www.numpy.org/) 
               - Pandas >0.13 (optional) (http://pandas.pydata.org/)
               - Matplotlib >1.4 (optional) (http://matplotlib.org/) 
               - CMF (optional) (http://fb09-pasig.umwelt.uni-giessen.de:8000/)
               - mpi4py (optional) (http://mpi4py.scipy.org/)
               - pathos (optional) (https://pypi.python.org/pypi/pathos/)
               :help: For specific questions, try to use the documentation website at:
http://fb09-pasig.umwelt.uni-giessen.de/spotpy/
For general things about parameter optimization techniques have a look at:
https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/
Pleas cite our paper, if you are using SPOTPY.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pandas
import copy
import numpy as np
'''
By B. Clausen, B.J.F. Biggs / Journal of Hydrology 237 (2000) 184-197

Uncertainty in hydrological signatures I. K. Westerberg and H. K. McMillan

'''

class HydroIndiciesError(Exception):
    """
    Define an own error class to know it is an error made by a hydroindex calculation to warn the use for wrong inputs
    """
    pass


def __isSorted(df):
    """
    If the pandas object is not sorted the comparision will failed with a valueError which will be caught 
    and noted as a unsorted list
    :param df: pandas datetime object 
    :return: bool
    """
    try:
        if sum(df == df.sort_values()) == df.__len__():
            return True
        else:
            return False
    except ValueError:
        return False


def __calcDev(a, b):
    """
    Calculate the relative error / derivation of two values
    If one parameter is zero the result is just 1, for example b = 0, so calculate: (a+0)/a = 1
    a ~= b iff result is ~zero [approximately]
    :param a: value a
    :type: float
    :param b: value b
    :type: float
    :return: relative error
    :type: float
    """
    if a != 0:
        return (a - b) / a
    elif b != 0:
        return (a - b) / b
    else:
        return 0

def __percentilwrapper(array, index):
    """
    Based to the definition of the paper with a 10-percentiles - 10% = 0.1 of the data are equal or less then the Q10 
    :array: data
    :type: list
    :index: percentil index
    :type: float / int
    :return: Numpy Percentil
    :rtype: float
    
    """
    return np.percentile(array, index)

def __calcMeanFlow(data):
    """
    Simply calculate the mean of the data
    :data: A list of float data
    :type: list
    :return: Mean
    :rtype: float
    """
    return np.mean(data)

def __calcMedianFlow(data):
    """
    Simply calculate the median (flow exceeded 50% of the time) of the data
    :data: A list of float data
    :type: list
    :return: Median
    :rtype: float
    """
    return np.percentile(data, 50)


def getMeanFlow(evaluation, simulation):
    """
    Simply calculate the mean of the data
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evaluation data
    :type: list
    
    :return: Mean
    :rtype: float
    
    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __calcMeanFlow(simulation)
    b = __calcMeanFlow(evaluation)
    return __calcDev(a,b)

def getMedianFlow(evaluation, simulation):
    """    
    Simply calculate the median (flow exceeded 50% of the time) of the data

    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evaluation data
    :type: list
    
    :return: Median
    :rtype: float
    
    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")
    a = __calcMedianFlow(simulation)
    b = __calcMedianFlow(evaluation)
    return __calcDev(a, b)

def getSkewness(evaluation, simulation):
    """
    Skewness, i.e. MF divided by Q50.
         
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evaluation data
    :type: list
    
    :return: derivation of the skewness
    :rtype: float
    
    """

    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __calcMeanFlow(simulation) / __calcMedianFlow(simulation)
    b = __calcMeanFlow(evaluation) / __calcMedianFlow(evaluation)
    return __calcDev(a, b)

def getCoeffVariation(evaluation, simulation):
    """
    
    Coefficient of variation, i.e. standard deviation divided by MF
    
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: derivation of the coefficient of variation
    :rtype: float

    """
    
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = np.std(simulation)/__calcMeanFlow(simulation)
    b = np.std(evaluation) / __calcMeanFlow(evaluation)
    return __calcDev(a, b)


def getQ001(evaluation, simulation):
    """
    The value of the 0.01 percentiles
    
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: derivation of the 0.01 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 0.01)
    b = __percentilwrapper(evaluation, 0.01)
    return __calcDev(a, b)

def getQ01(evaluation, simulation):
    """
    The value of the 0.1 percentiles
    
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: derivation of the 0.1 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 0.1)
    b = __percentilwrapper(evaluation, 0.1)
    return __calcDev(a, b)

def getQ1(evaluation, simulation):
    """
    The value of the 1 percentiles
    
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: derivation of the 1 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 1)
    b = __percentilwrapper(evaluation, 1)
    return __calcDev(a, b)

def getQ5(evaluation, simulation):
    """
    The value of the 5 percentiles
    
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: derivation of the 5 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 5)
    b = __percentilwrapper(evaluation, 5)
    return __calcDev(a, b)

def getQ10(evaluation, simulation):
    """
    The value of the 10 percentiles
    
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: derivation of the 10 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 10)
    b = __percentilwrapper(evaluation, 10)
    return __calcDev(a, b)

def getQ20(evaluation, simulation):
    """
    The value of the 20 percentiles
    
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: derivation of the 20 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 20)
    b = __percentilwrapper(evaluation, 20)
    return __calcDev(a, b)

def getQ85(evaluation, simulation):
    """
    The value of the 85 percentiles
    
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: derivation of the 85 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 85)
    b = __percentilwrapper(evaluation, 85)
    return __calcDev(a, b)

def getQ95(evaluation, simulation):
    """
    The value of the 95 percentiles
    
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: derivation of the 95 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 95)
    b = __percentilwrapper(evaluation, 95)
    return __calcDev(a, b)

def getQ99(evaluation, simulation):
    """
    The value of the 99 percentiles
    
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: derivation of the 99 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 99)
    b = __percentilwrapper(evaluation, 99)
    return __calcDev(a, b)


def getDuration(evaluation, simulation,datetime_series,key):
    """
    Get high and low-flow yearly-average event duration which have a threshold of [0.2, 1,3,5,7,9] the median
    
    
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :datetime_series: a pandas data object with sorted (may not be complete but sorted) dates 
    :type: pandas datetime
    
    :key: which threshold calculation should be used. Allowed keys are: [0.2, 1,3,5,7,9] 
    :type: int/float
    
    :return: mean of deviation of average duration of a year
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroIndiciesError("Simulation / observation data and the datetime_series have not the same length")


    FRE_a, rawDUR_a, DUR_a = __calcFRE(simulation,datetime_series)
    FRE_b, rawDUR_b, DUR_b = __calcFRE(evaluation,datetime_series)

    sum_dev = 0.0
    for y in DUR_a:
        if key in DUR_a[y]:
            sum_dev += np.abs(__calcDev(DUR_a[y][key], DUR_b[y][key]))
        else:
            HydroIndiciesError("Key " + str(key) + " does not exists")

    return sum_dev / DUR_a.__len__()


def getFrequency(evaluation, simulation, datetime_series, key):
    """
    Get high and low-flow event frequencies which have a threshold of [0.2, 1,3,5,7,9] the median
    
        
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :datetime_series: a pandas data object with sorted (may not be complete but sorted) dates 
    :type: pandas datetime
    
    :key: which threshold calculation should be used. Allowed keys are: [0.2, 1,3,5,7,9] 
    :type: int/float
    
    :return: mean of deviation of average frequency of a year
    :rtype: float

    
    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroIndiciesError("Simulation / observation data and the datetime_series have not the same length")

    FRE_a, rawDUR_a, DUR_a = __calcFRE(simulation, datetime_series)
    FRE_b, rawDUR_b, DUR_b = __calcFRE(evaluation, datetime_series)

    sum_dev = 0.0
    for y in FRE_a:
        if key in FRE_a[y]:
            sum_dev += np.abs(__calcDev(FRE_a[y][key], FRE_b[y][key]))
        else:
            raise HydroIndiciesError("Key " + str(key) + " does not exists")

    return sum_dev / FRE_a.__len__()


def __calcFRE(dailyflows,datetime_series):
    """
    High  ̄flow frequency, i.e. average number of high  ̄flow events per year, using a threshold of 
        0.2 times the median
        1 times the median
        3   -""-        
        5   -""-
        7   -""-
        9   -""-
    
    :param dailyflows: list of float
    :param datetime_series: pandas datetime object
    :return: 
    """

    Count_per_year = {}
    duration_per_year = {}
    tmp_duration_logger_per_year = {}
    threshold_on_per_year = {}
    tmpStdDurLG = {'start': "0000-00-00", 'end': '0000-00-00', 'duration': 0}
    index = 0
    tmp_duration_logger_per_year_helper = {}

    for j in [0.2, 1, 3, 5, 7, 9]:
        tmp_duration_logger_per_year_helper[j] = copy.deepcopy(tmpStdDurLG)

    if __isSorted(datetime_series):
        for d in datetime_series:
            yr = d.to_pydatetime().year
            if yr not in Count_per_year:
                # Define a bunch of arrays to handle, save and organize the analyze of the data as most as possible at the same time
                Count_per_year[yr] = {0.2:0, 1: 0, 3: 0, 5: 0, 7: 0, 9: 0}
                duration_per_year[yr] = {0.2:[], 1: [], 3: [], 5: [], 7: [], 9: []}
                tmp_duration_logger_per_year[yr] = tmp_duration_logger_per_year_helper
                threshold_on_per_year[yr] = {0.2: False, 1: False, 3: False, 5: False, 7: False, 9: False}
                # And save the old years duration object:
                if index > 0:
                    tmp_lastyear = datetime_series[index - 1].to_pydatetime().year
                    for j in [0.2, 1, 3, 5, 7, 9]:
                        if tmp_duration_logger_per_year[tmp_lastyear][j]["duration"] > 0:
                            tmp_duration_logger_per_year[tmp_lastyear][j]["end"] = datetime_series[
                                index - 1].to_pydatetime().strftime("%Y-%m-%d")
                            duration_per_year[tmp_lastyear][j].append(
                                copy.deepcopy(tmp_duration_logger_per_year[tmp_lastyear][j]))
                            tmp_duration_logger_per_year[tmp_lastyear][j] = copy.deepcopy(tmpStdDurLG)

            for j in [0.2, 1, 3, 5, 7, 9]:

                if dailyflows[index] > j * __calcMedianFlow(dailyflows):
                    Count_per_year[yr][j] += 1
                    if not threshold_on_per_year[yr][j]:
                        threshold_on_per_year[yr][j] = True
                        tmp_duration_logger_per_year[yr][j]["start"] = d.to_pydatetime().strftime("%Y-%m-%d")
                    tmp_duration_logger_per_year[yr][j]["duration"] = tmp_duration_logger_per_year[yr][j][
                                                                          "duration"] + 1
                else:
                    if threshold_on_per_year[yr][j]:
                        threshold_on_per_year[yr][j] = False
                        tmp_duration_logger_per_year[yr][j]["end"] = datetime_series[
                            index - 1].to_pydatetime().strftime("%Y-%m-%d")
                        if tmp_duration_logger_per_year[yr][j]["duration"] > 0:
                            # Here we save the logged flood into the bug array
                            duration_per_year[yr][j].append(copy.deepcopy(tmp_duration_logger_per_year[yr][j]))

                        tmp_duration_logger_per_year[yr][j] = copy.deepcopy(tmpStdDurLG)

                    else:
                        threshold_on_per_year[yr][j] = False
            index += 1
    else:
        raise HydroIndiciesError("The timeseries is not sorted, so a calculation can not be performed")

    DUR = {}
    for i in duration_per_year:
        DUR[i] = {}
        for du in duration_per_year[i]:
            DUR[i][du] = duration_per_year[i][du].__len__()

    return Count_per_year, duration_per_year, DUR


def getLowFlowVar(evaluation, simulation,datetime_series):
    """
    
    Mean of annual minimum flow divided by the median flow (Jowett and Duncan, 1990)
        
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :datetime_series: a pandas data object with sorted (may not be complete but sorted) dates 
    :type: pandas datetime
    
    
    :return: mean of deviation of the low flow variation
    :rtype: float

    """
     
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroIndiciesError("Simulation / observation data and the datetime_series have not the same length")
    if not __isSorted(datetime_series):
        raise HydroIndiciesError("datetime_series data are not sorted")

    sim_LFV = __calcAnnularData(simulation,datetime_series,"min") / __calcMedianFlow(simulation)
    obs_LFV = __calcAnnularData(evaluation, datetime_series, "min") / __calcMedianFlow(evaluation)
    return __calcDev(sim_LFV,obs_LFV)


def getHighFlowVar(evaluation, simulation, datetime_series):
    """
    Mean of annual maximum flow divided by the median flow (Jowett and Duncan, 1990)

    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evaluation data
    :type: list
    
    :datetime_series: a pandas data object with sorted (may not be complete but sorted) dates 
    :type: pandas datetime
    
    
    :return: mean of deviation of the high flow variation
    :rtype: float

    """
    
    
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroIndiciesError("Simulation / observation data and the datetime_series have not the same length")
    if not __isSorted(datetime_series):
        raise HydroIndiciesError("datetime_series data are not sorted")

    sim_LFV = __calcAnnularData(simulation, datetime_series, "max") / __calcMedianFlow(simulation)
    obs_LFV = __calcAnnularData(evaluation, datetime_series, "max") / __calcMedianFlow(evaluation)
    return __calcDev(sim_LFV, obs_LFV)


def __calcAnnularData(data,datetime_series,what):
    """
    
    :param data: float list 
    :param datetime_series: sorted pandas date time object
    :param what: string which switches the calculation method. Allowed are:
        "min": the minimum value of a year
        "max": the maximum value of a year
    :return: float - mean of min/max per year
    """
    data_per_year_tmp=[]
    data_per_year={}
    index=0
    for d in datetime_series:
        yr = d.to_pydatetime().year
        if yr not in data_per_year:
            if index > 0:
                data_per_year[datetime_series[index-1].to_pydatetime().year] = data_per_year_tmp
                data_per_year[datetime_series[index-1].to_pydatetime().year] = data_per_year_tmp
                data_per_year_tmp = []
        data_per_year_tmp.append(data[index])
        index+=1

    summarized = []
    if what == "min":
        for y in data_per_year:
            summarized.append(np.min(data_per_year[y]))
        return np.mean(summarized)
    elif what == "max":
        for y in data_per_year:
            summarized.append(np.max(data_per_year[y]))
        return np.mean(summarized)
    else:
        raise HydroIndiciesError("The parameter what="+what+" is not defined")


def getBaseflowIndex(evaluation, simulation, datetime_series):
    """
    We may have to use baseflow devided with total discharge
    How could we do that?
    https://de.wikipedia.org/wiki/Niedrigwasser
    See also http://people.ucalgary.ca/~hayashi/kumamoto_2014/lectures/2_3_baseflow.pdf for formular
    
    I would propose:
    discharge: minimum water flow
    basic: Q50?
    
    Look at: IH_108.pdf
    
    
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evaluation data
    :type: list
    
    :datetime_series: a pandas data object with sorted (may not be complete but sorted) dates 
    :type: pandas datetime
    
    :return: deviation of base flow index
    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroIndiciesError("Simulation / observation data and the datetime_series have not the same length")
    if not __isSorted(datetime_series):
        raise HydroIndiciesError("datetime_series data are not sorted")

    bfi_sim = __calcBaseflowIndex(simulation,datetime_series)
    bfi_obs = __calcBaseflowIndex(evaluation,datetime_series)
    sum_sim = 0.0
    sum_obs = 0.0

    for y in bfi_obs:
        sum_obs+=bfi_obs[y]
    for y in bfi_sim:
        sum_sim+=bfi_sim[y]

    sum_obs = sum_obs/bfi_obs.__len__()
    sum_sim = sum_sim / bfi_sim.__len__()

    return __calcDev(sum_sim,sum_obs)


def __calcBaseflowIndex(data,datetime_series):
    """
    .. :math:
        "BF/ TD" where BF is the median of the data and TD the minimum of the data per year
    :param data: float list
    :param datetime_series: sorted pandas daetime objext
    :return: dict of BFI per year 
    """
    Min_per_year={}
    Q50_per_year = {}
    data_per_year_tmp=[]
    index=0
    for d in datetime_series:
        yr = d.to_pydatetime().year
        if yr not in Min_per_year:
            if index > 0:
                Min_per_year[datetime_series[index-1].to_pydatetime().year] = np.min(data_per_year_tmp)
                Q50_per_year[datetime_series[index-1].to_pydatetime().year] = np.median(data_per_year_tmp)
                data_per_year_tmp = []
        data_per_year_tmp.append(data[index])
        index+=1

    BFI = {}
    for y in Min_per_year:
        BFI[y]=Q50_per_year[y]/Min_per_year[y]
    return BFI


def getSlopeFDC(evaluation, simulation):
    """
    Slope of the FDC between the 33 and 66 % exceedance values of streamflow normalised by its mean (Yadav et al., 2007)
    
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evaluation data
    :type: list
    
    :return: deviation of the slope
    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    return __calcDev(__calcSlopeFDC(simulation),__calcSlopeFDC(evaluation))


def __calcSlopeFDC(data):
    """
    The main idea is to use a threshold by the mean of the data and use the first occurrence of a 33% exceed and a 66% 
    exceed and calculate the factor of how many times is the 66% exceed higher then the 33% exceed.
    If 33% or 66% exceed does not exists then just give 0 back for a slope of 0 (horizontal line) 
    :param data: float list 
    :return: float slope
    """
    upper33_data = np.sort(data)[np.sort(data) >= 1.33 * __calcMeanFlow(data)]
    upper66_data = np.sort(data)[np.sort(data) >= 1.66 * __calcMeanFlow(data)]
    if upper33_data.__len__() > 0 and upper66_data.__len__() > 0:
        if upper66_data[0] != 0:
            return upper33_data[0] / upper66_data[0]
        else:
            return 0.0
    else:
        return 0.0


def __help():
    print("Use .__doc__ to see description of every function")