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
from spotpy.examples.spot_setup_hymod import spot_setup
import pandas,pprint
import copy
import numpy as np
'''
By B. Clausen, B.J.F. Biggs / Journal of Hydrology 237 (2000) 184-197

Uncertainty in hydrological signatures I. K. Westerberg and H. K. McMillan

'''

class HydroIndiciesError(Exception):
    pass


def __isSorted(df):
    try:
        if sum(df == df.sort_values()) == df.__len__():
            return True
        else:
            return False
    except ValueError:
        return False


def __calcDev(a, b):
    """We calculate the relative error"""
    if a != 0:
        return (a - b) / a
    elif b != 0:
        return (a - b) / b
    else:
        return 0

def __percentilwrapper(array, index):
    """Based to the definition of the paper with a 10-percentiles - 10% = 0.1 of the data are equal or less then the Q10 """
    return np.percentile(array, index)

def __calcMeanFlow(data):
    """Simply calculate the mean of the data"""
    return np.mean(data)

def __calcMedianFlow(data):
    """Simply calculate the median (flow exceeded 50% of the time) of the data"""
    return np.percentile(data, 50)


def getMeanFlow(simulations,observations):
    """Simply calculate the mean of the data"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __calcMeanFlow(simulations)
    b = __calcMeanFlow(observations)
    return __calcDev(a,b)

def getMedianFlow(simulations, observations):
    """Simply calculate the median (flow exceeded 50% of the time) of the data"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")
    a = __calcMedianFlow(simulations)
    b = __calcMedianFlow(observations)
    return __calcDev(a, b)

def getSkewness(simulations, observations):
    """Skewness, i.e. MF divided by Q50."""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __calcMeanFlow(simulations) / __calcMedianFlow(simulations)
    b = __calcMeanFlow(observations) / __calcMedianFlow(observations)
    return __calcDev(a, b)

def getCoeffVariation(simulations, observations):
    """Coefficient of variation, i.e. standard deviation divided by MF"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = np.std(simulations)/__calcMeanFlow(simulations)
    b = np.std(observations) / __calcMeanFlow(observations)
    return __calcDev(a, b)


def getQ001(simulations, observations):
    """The value of the 0.01 percentiles (german: quantil)"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulations, 0.01)
    b = __percentilwrapper(observations, 0.01)
    return __calcDev(a, b)

def getQ01(simulations, observations):
    """The value of the 0.1 percentiles (german: quantil)"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulations, 0.1)
    b = __percentilwrapper(observations, 0.1)
    return __calcDev(a, b)

def getQ1(simulations, observations):
    """The value of the 1 percentiles (german: quantil)"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulations, 1)
    b = __percentilwrapper(observations, 1)
    return __calcDev(a, b)

def getQ5(simulations, observations):
    """The value of the 5 percentiles (german: quantil)"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulations, 5)
    b = __percentilwrapper(observations, 5)
    return __calcDev(a, b)

def getQ10(simulations, observations):
    """The value of the 10 percentiles (german: quantil)"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulations, 10)
    b = __percentilwrapper(observations, 10)
    return __calcDev(a, b)

def getQ20(simulations, observations):
    """The value of the 20 percentiles (german: quantil)"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulations, 20)
    b = __percentilwrapper(observations, 20)
    return __calcDev(a, b)

def getQ85(simulations, observations):
    """The value of the 85 percentiles (german: quantil)"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulations, 85)
    b = __percentilwrapper(observations, 85)
    return __calcDev(a, b)

def getQ95(simulations, observations):
    """The value of the 95 percentiles (german: quantil)"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulations, 95)
    b = __percentilwrapper(observations, 95)
    return __calcDev(a, b)

def getQ99(simulations, observations):
    """The value of the 99 percentiles (german: quantil)"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulations, 99)
    b = __percentilwrapper(observations, 99)
    return __calcDev(a, b)


def getDuration(simulations, observations,datetime_series,key):
    """Get high and low-flow yearly-average event duration which have a threshold of [0.2, 1,3,5,7,9] the median"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    if simulations.__len__() != datetime_series.__len__():
        raise HydroIndiciesError("Simulation / observation data and the datetime_series have not the same length")


    FRE_a, rawDUR_a, DUR_a = __calcFRE(simulations,datetime_series)
    FRE_b, rawDUR_b, DUR_b = __calcFRE(observations,datetime_series)

    sum_dev = 0.0
    for y in DUR_a:
        if key in DUR_a[y]:
            sum_dev += np.abs(__calcDev(DUR_a[y][key], DUR_b[y][key]))
        else:
            HydroIndiciesError("Key " + str(key) + " does not exists")

    return sum_dev / DUR_a.__len__()


def getFrequency(simulations, observations, datetime_series, key):
    """Get high and low-flow event frequencies which have a threshold of [0.2, 1,3,5,7,9] the median"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    if simulations.__len__() != datetime_series.__len__():
        raise HydroIndiciesError("Simulation / observation data and the datetime_series have not the same length")

    FRE_a, rawDUR_a, DUR_a = __calcFRE(simulations, datetime_series)
    FRE_b, rawDUR_b, DUR_b = __calcFRE(observations, datetime_series)

    sum_dev = 0.0
    for y in FRE_a:
        if key in FRE_a[y]:
            sum_dev += np.abs(__calcDev(FRE_a[y][key], FRE_b[y][key]))
        else:
            raise HydroIndiciesError("Key " + str(key) + " does not exists")

    return sum_dev / FRE_a.__len__()


def __calcFRE(dailyflows,datetime_series):
    """High  ̄flow frequency, i.e. average number of high  ̄flow events per year, using a threshold of 
        0.2 times the median
        1 times the median
        3   -""-        
        5   -""-
        7   -""-
        9   -""-
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


def getLowFlowVar(simulations, observations,datetime_series):
    """Mean of annual minimum flow divided by the median flow (Jowett and Duncan, 1990)"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    if simulations.__len__() != datetime_series.__len__():
        raise HydroIndiciesError("Simulation / observation data and the datetime_series have not the same length")
    if not __isSorted(datetime_series):
        raise HydroIndiciesError("datetime_series data are not sorted")

    sim_LFV = __calcAnnularData(simulations,datetime_series,"min") / __calcMedianFlow(simulations)
    obs_LFV = __calcAnnularData(observations, datetime_series, "min") / __calcMedianFlow(observations)
    return __calcDev(sim_LFV,obs_LFV)


def getHighFlowVar(simulations, observations, datetime_series):
    """Mean of annual minimum flow divided by the median flow (Jowett and Duncan, 1990)"""
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    if simulations.__len__() != datetime_series.__len__():
        raise HydroIndiciesError("Simulation / observation data and the datetime_series have not the same length")
    if not __isSorted(datetime_series):
        raise HydroIndiciesError("datetime_series data are not sorted")

    sim_LFV = __calcAnnularData(simulations, datetime_series, "max") / __calcMedianFlow(simulations)
    obs_LFV = __calcAnnularData(observations, datetime_series, "max") / __calcMedianFlow(observations)
    return __calcDev(sim_LFV, obs_LFV)


def __calcAnnularData(data,datetime_series,what):
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


def getBaseflowIndex(simulations, observations, datetime_series):
    """
    We may have to use baseflow devided with total discharge
    How could we do that?
    https://de.wikipedia.org/wiki/Niedrigwasser
    See also http://people.ucalgary.ca/~hayashi/kumamoto_2014/lectures/2_3_baseflow.pdf for formular
    
    I would propose:
    discharge: minimum water flow
    basic: Q50?
    
    Look at: IH_108.pdf
    :return: 
    """
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    if simulations.__len__() != datetime_series.__len__():
        raise HydroIndiciesError("Simulation / observation data and the datetime_series have not the same length")
    if not __isSorted(datetime_series):
        raise HydroIndiciesError("datetime_series data are not sorted")

    bfi_sim = __calcBaseflowIndex(simulations,datetime_series)
    bfi_obs = __calcBaseflowIndex(observations,datetime_series)
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


def getSlopeFDC(simulations, observations):
    """
    Slope of the FDC between the 33 and 66 % exceedance values of streamflow normalised by its mean (Yadav et al., 2007)
    """
    if simulations.__len__() != observations.__len__():
        raise HydroIndiciesError("Simulation and observation data have not the same length")

    return __calcDev(__calcSlopeFDC(simulations),__calcSlopeFDC(observations))


def __calcSlopeFDC(data):
    upper33_data = np.sort(data)[np.sort(data) >= 1.33 * __calcMeanFlow(data)]
    upper66_data = np.sort(data)[np.sort(data) >= 1.66 * __calcMeanFlow(data)]
    if upper33_data.__len__() > 0 and upper66_data.__len__() > 0:
        if upper66_data[0] != 0:
            return upper33_data[0] / upper66_data[0]
        else:
            return 1
    else:
        return 1


def __help():
    print("Use .__doc__ to see description of every function")