# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Benjamin Manns

This tool holds functions for caclulation of hyrological signatures. It takes Python-lists of simulation and observation data
returns the hydrological signature value of interest.

The code is based on:

B. Clausen, B.J.F. Biggs / Journal of Hydrology 237 (2000) 184-197 Flow variables for ecological studies in temperate streams: groupings based on covariance
I. K. Westerberg and H. K. McMillan / HESS 19 (2015) 3951-3968 Uncertainty in hydrological signatures
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import numpy as np
import datetime
import warnings


class HydroSignaturesError(Exception):
    """
    Define an own error class to know it is an error made by a hydroindex calculation to warn the use for wrong inputs
    """
    pass


class SuitableInput:
    def __init__(self, datm, section):
        """
        Checks whether the date time series suits to a chosen section (year, month, day, hour). So if we may have daily
         data, a hourly section may not work properly. All of this inappropriate choices generate a warning

        :param datm:
        :type datm: pandas datetime object
        :param section: section in [year, month, day, hour]
        :type section: string
        """
        self.datm = datm
        self.section = section
        b, r = self.__calc()
        if not b:
            warnings.warn("\nYour chose section was [" + self.section + "] and this is not suitable to you time data.\n"
                          "Your time data have an interval of [" + str(r) + " " + self.section + "]")

    def __calc(self):
        if self.datm.__len__() > 1:
            diff = (self.datm[1].to_pydatetime() - self.datm[0].to_pydatetime()).total_seconds()
            if self.section == "year":
                return diff / (3600 * 24 * 365) <= 1.0, diff / (3600 * 24 * 365)
            elif self.section == "month":
                return diff / (3600 * 24 * 30) <= 1.0, diff / (3600 * 24 * 30)
            elif self.section == "day":
                return diff / (3600 * 24) <= 1.0, diff / (3600 * 24)
            elif self.section == "hour":
                return diff <= 3600, diff / 3600
            else:
                raise Exception("The section [" + self.section + "] is not defined in "+str(self))


def __isSorted(df):
    """
    If the pandas object is not sorted the comparision will failed with a ValueError which will be caught
    and noted as a unsorted list

    :param df: time series
    :type df: pandas datetime object
    :return: if the pandas object is sorted
    :rtype: bool

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
    If one parameter is zero the result is just 1, for example b = 0, so calculate:
    :math:`\\frac{a+0}{a} = 1` and also
    :math:`a =  b  \\Leftrightarrow  return =  0` [approximately]

    :param a: Value a
    :type a: float
    :param b: Value b
    :type b: float
    :return: relative error of a and b (numeric definition)
    :rtype: float
    """
    if a != 0:
        return (a - b) / a
    elif b != 0:
        return (a - b) / b
    else:
        return 0


def __percentilwrapper(array, index):
    """
    A Percentil Wrapper to have a easy chance to modify the percentil calculation for the following functions

    :param array: float array
    :type array: list
    :param index: which percentil should be used
    :type index: int
    :return: the percentil
    :rtype: float
    """
    return np.percentile(array, index)


def __calcMeanFlow(data):
    """
    Simply calculate the mean of the data

    :param data: A list of float data
    :type data: list
    :return: Mean
    :rtype: float
    """
    return np.mean(data)


def __calcMedianFlow(data):
    """
    Simply calculate the median (flow exceeded 50% of the time) of the data

    :param data: A list of float data
    :type data: list
    :return: Median
    :rtype: float
    """
    return np.percentile(data, 50)


def getMeanFlow(evaluation, simulation):
    """
    Simply calculate the mean of the data

    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    
    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list
    
    :return: Mean
    :rtype: float
    
    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = __calcMeanFlow(simulation)
    b = __calcMeanFlow(evaluation)
    return __calcDev(a, b)


def getMedianFlow(evaluation, simulation):
    """    
    Simply calculate the median (flow exceeded 50% of the time) of the data

    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    
    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list
    
    :return: Median
    :rtype: float
    
    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")
    a = __calcMedianFlow(simulation)
    b = __calcMedianFlow(evaluation)
    return __calcDev(a, b)


def getSkewness(evaluation, simulation):
    """
    Skewness, i.e. the mean flow data divided by Q50 (50 percentil / median flow) .
         
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list

    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list
    
    :return: derivation of the skewness
    :rtype: float
    
    """

    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = __calcMeanFlow(simulation) / __calcMedianFlow(simulation)
    b = __calcMeanFlow(evaluation) / __calcMedianFlow(evaluation)
    return __calcDev(a, b)


def getCoeffVariation(evaluation, simulation):
    """
    
    Coefficient of variation, i.e. standard deviation divided by mean flow

    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list

    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list

    :return: derivation of the coefficient of variation
    :rtype: float

    """

    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = np.std(simulation) / __calcMeanFlow(simulation)
    b = np.std(evaluation) / __calcMeanFlow(evaluation)
    return __calcDev(a, b)


def getQ001(evaluation, simulation):
    """
    The value of the 0.01 percentiles
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list

    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list

    :return: derivation of the 0.01 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 0.01)
    b = __percentilwrapper(evaluation, 0.01)
    return __calcDev(a, b)


def getQ01(evaluation, simulation):
    """
    The value of the 0.1 percentiles
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list

    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list

    :return: derivation of the 0.1 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 0.1)
    b = __percentilwrapper(evaluation, 0.1)
    return __calcDev(a, b)


def getQ1(evaluation, simulation):
    """
    The value of the 1 percentiles
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list

    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list

    :return: derivation of the 1 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 1)
    b = __percentilwrapper(evaluation, 1)
    return __calcDev(a, b)


def getQ5(evaluation, simulation):
    """
    The value of the 5 percentiles
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list

    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list

    :return: derivation of the 5 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 5)
    b = __percentilwrapper(evaluation, 5)
    return __calcDev(a, b)


def getQ10(evaluation, simulation):
    """
    The value of the 10 percentiles
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list

    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list

    :return: derivation of the 10 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 10)
    b = __percentilwrapper(evaluation, 10)
    return __calcDev(a, b)


def getQ20(evaluation, simulation):
    """
    The value of the 20 percentiles
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list

    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list

    :return: derivation of the 20 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 20)
    b = __percentilwrapper(evaluation, 20)
    return __calcDev(a, b)


def getQ85(evaluation, simulation):
    """
    The value of the 85 percentiles
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list

    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list

    :return: derivation of the 85 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 85)
    b = __percentilwrapper(evaluation, 85)
    return __calcDev(a, b)


def getQ95(evaluation, simulation):
    """
    The value of the 95 percentiles
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list

    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list

    :return: derivation of the 95 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 95)
    b = __percentilwrapper(evaluation, 95)
    return __calcDev(a, b)


def getQ99(evaluation, simulation):
    """
    The value of the 99 percentiles
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list

    :param simulation: Simulated data to compared with evaluation data
    :type simulation: list

    :return: derivation of the 99 percentiles
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    a = __percentilwrapper(simulation, 99)
    b = __percentilwrapper(evaluation, 99)
    return __calcDev(a, b)


def getAverageFloodOverflowPerSection(evaluation, simulation, datetime_series, threshold_factor=3, section="day"):
    """
    All measurements are scanned where there are overflow events. Based on the section we summarize events per year, 
    month, day, hour.
    Of course we need a datetime_series which has the the suitable resolution. So, for example, if you group the 
    overflow events hourly but you have only daily data it the function will work but not very useful.
    
    However for every section the function collect the overflow value, i.e. value - threshold and calc the deviation
    of the means of this overflows.
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_factor: which times the median we use for a threshold
    :type threshold_factor: float
    :param section: one of ["year","month","day","hour"]
    :type section: string
    :return: deviation of means of overflow value
    :rtype: float
    """



    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    DUR_a = __calcFloodDuration(simulation, datetime_series, threshold_factor, section,"flood")
    DUR_b = __calcFloodDuration(evaluation, datetime_series, threshold_factor, section,"flood")


    for_mean_a = []
    for_mean_b = []
    for y in DUR_a:
        if DUR_a[y].__len__() > 0:
            for elem in range(DUR_a[y].__len__()):
                for ov in DUR_a[y][elem]["overflow"]:
                    for_mean_a.append(ov)
        for y in DUR_b:
            if DUR_b[y].__len__() > 0:
                for elem in range(DUR_b[y].__len__()):
                    for ov in DUR_b[y][elem]["overflow"]:
                        for_mean_b.append(ov)

    return __calcDev(np.mean(for_mean_a),np.mean(for_mean_b))


def getAverageFloodFrequencyPerSection(evaluation, simulation, datetime_series, threshold_factor=3, section="day"):
    """
    This function calculates the average frequency per every section in the given interval of the datetime_series. 
    So if the datetime is recorded all 5 min we use this fine intervall to count all records which are in flood.
     
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_factor: which times the median we use for a threshold
    :type threshold_factor: float
    :param section: one of ["year","month","day","hour"]
    :type section: string
    :return: deviation of means of flood frequency per section
    :rtype: float
    """

    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    DUR_a = __calcFloodDuration(simulation, datetime_series, threshold_factor, section,"flood")
    DUR_b = __calcFloodDuration(evaluation, datetime_series, threshold_factor, section,"flood")

    sum_dev = 0.0

    for y in DUR_a:
        sum_dur_1 = 0.0
        sum_dur_2 = 0.0
        for elem in DUR_a[y]:
            sum_dur_1 += elem["duration"]
        for elem in DUR_b[y]:
            sum_dur_2 += elem["duration"]

        sum_dev += __calcDev(sum_dur_1, sum_dur_2)

    return sum_dev / DUR_a.__len__()


def getAverageFloodDuration(evaluation, simulation, datetime_series, threshold_factor=3, section="day"):
    """
    Get high and low-flow yearly-average event duration which have a threshold of [0.2, 1,3,5,7,9] the median
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_factor: which times the median we use for a threshold
    :type threshold_factor: float
    :param section: one of ["year","month","day","hour"]
    :type section: string
    :return: deviation of means of flood durations
    :rtype: float
    """

    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    DUR_a = __calcFloodDuration(simulation, datetime_series, threshold_factor, section,"flood")
    DUR_b = __calcFloodDuration(evaluation, datetime_series, threshold_factor, section,"flood")

    sum_dev = 0.0
    for y in DUR_a:
        sum_diff_1 = 0.0
        sum_diff_2 = 0.0
        if DUR_a[y].__len__() > 0:
            for elem in range(DUR_a[y].__len__()):
                d_start_a = datetime.datetime.strptime(DUR_a[y][elem]["start"], "%Y-%m-%d %H:%M:%S")
                d_end_a = datetime.datetime.strptime(DUR_a[y][elem]["end"], "%Y-%m-%d %H:%M:%S")
                if d_end_a.date() == d_start_a.date():
                    sum_diff_1 += 24 * 3600
                else:
                    d_diff_a = d_end_a - d_start_a
                    sum_diff_1 += d_diff_a.seconds
            sum_diff_av_1 = sum_diff_1 / DUR_a[y].__len__()
        else:
            sum_diff_av_1 = 0

        if DUR_b[y].__len__() > 0:
            for elem in range(DUR_b[y].__len__()):
                d_start_b = datetime.datetime.strptime(DUR_b[y][elem]["start"], "%Y-%m-%d %H:%M:%S")
                d_end_b = datetime.datetime.strptime(DUR_b[y][elem]["end"], "%Y-%m-%d %H:%M:%S")
                if d_end_b.date() == d_start_b.date():
                    d_diff_b = datetime.timedelta(1)
                else:
                    d_diff_b = d_end_b - d_start_b
                sum_diff_2 += d_diff_b.seconds
                # print(sum_diff_2)
            sum_diff_av_2 = sum_diff_2 / DUR_b[y].__len__()
        else:
            sum_diff_av_2 = 0

        # print(str(sum_diff_av_2) +"|"+ str(sum_diff_av_1))
        if section == "year":
            sum_dev += __calcDev(sum_diff_av_1 / (365 * 24 * 3600),
                                 sum_diff_av_2 / (365 * 24 * 3600))
        elif section == "month":
            sum_dev += __calcDev(sum_diff_av_1 / (30 * 24 * 3600),
                                 sum_diff_av_2 / (30 * 24 * 3600))
        elif section == "day":
            sum_dev += __calcDev(sum_diff_av_1 / (24 * 3600),
                                 sum_diff_av_2 / (24 * 3600))
        elif section == "hour":
            sum_dev += __calcDev(sum_diff_av_1 / (3600), sum_diff_av_2 / (3600))
        else:
            raise HydroSignaturesError("Your section: " + section + " is not valid. See pydoc of this function")

    return sum_dev / DUR_a.__len__()


def getAverageBaseflowUnderflowPerSection(evaluation, simulation, datetime_series, threshold_factor=3, section="day"):
    """
    All measurements are scanned where there are overflow events. Based on the section we summarize events per year, 
    month, day, hour.
    Of course we need a datetime_series which has the the suitable resolution. So, for example, if you group the 
    overflow events hourly but you have only daily data it the function will work but not very useful.

    However for every section the function collect the overflow value, i.e. value - threshold  and calc the deviation 
    of the means of this overflows.

    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_factor: which times the median we use for a threshold
    :type threshold_factor: float
    :param section: one of ["year","month","day","hour"]
    :type section: string
    :return: deviation of means of underflow value
    :rtype: float

    """

    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    DUR_a = __calcFloodDuration(simulation, datetime_series, threshold_factor, section,"baseflow")
    DUR_b = __calcFloodDuration(evaluation, datetime_series, threshold_factor, section,"baseflow")

    for_mean_a = []
    for_mean_b = []
    for y in DUR_a:
        if DUR_a[y].__len__() > 0:
            for elem in range(DUR_a[y].__len__()):
                for ov in DUR_a[y][elem]["underflow"]:
                    for_mean_a.append(ov)
        for y in DUR_b:
            if DUR_b[y].__len__() > 0:
                for elem in range(DUR_b[y].__len__()):
                    for ov in DUR_b[y][elem]["underflow"]:
                        for_mean_b.append(ov)

    return __calcDev(np.mean(for_mean_a), np.mean(for_mean_b))


def getAverageBaseflowFrequencyPerSection(evaluation, simulation, datetime_series, threshold_factor=3, section="day"):
    """
    This function calculates the average frequency per every section in the given interval of the datetime_series. 
    So if the datetime is recorded all 5 min we use this fine intervall to count all records which are in flood.

    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_factor: which times the median we use for a threshold
    :type threshold_factor: float
    :param section: one of ["year","month","day","hour"]
    :type section: string
    :return: deviation of means of baseflow frequency per section
    :rtype: float
    """

    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    DUR_a = __calcFloodDuration(simulation, datetime_series, threshold_factor, section,"baseflow")
    DUR_b = __calcFloodDuration(evaluation, datetime_series, threshold_factor, section,"baseflow")

    sum_dev = 0.0

    for y in DUR_a:
        sum_dur_1 = 0.0
        sum_dur_2 = 0.0
        for elem in DUR_a[y]:
            sum_dur_1 += elem["duration"]
        for elem in DUR_b[y]:
            sum_dur_2 += elem["duration"]

        sum_dev += __calcDev(sum_dur_1, sum_dur_2)

    return sum_dev / DUR_a.__len__()


def getAverageBaseflowDuration(evaluation, simulation, datetime_series, threshold_factor=3, section="day"):
    """
    Get high and low-flow yearly-average event duration which have a threshold of threshold_factor the median

    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_factor: which times the median we use for a threshold
    :type threshold_factor: float
    :param section: one of ["year","month","day","hour"]
    :type section: string
    :return: deviation of means of baseflow duration
    :rtype: float

    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    DUR_a = __calcFloodDuration(simulation, datetime_series, threshold_factor, section,"baseflow")
    DUR_b = __calcFloodDuration(evaluation, datetime_series, threshold_factor, section,"baseflow")

    sum_dev = 0.0
    for y in DUR_a:
        sum_diff_1 = 0.0
        sum_diff_2 = 0.0
        if DUR_a[y].__len__() > 0:
            for elem in range(DUR_a[y].__len__()):
                d_start_a = datetime.datetime.strptime(DUR_a[y][elem]["start"], "%Y-%m-%d %H:%M:%S")
                d_end_a = datetime.datetime.strptime(DUR_a[y][elem]["end"], "%Y-%m-%d %H:%M:%S")
                if d_end_a.date() == d_start_a.date():
                    sum_diff_1 += 24 * 3600
                else:
                    d_diff_a = d_end_a - d_start_a
                    sum_diff_1 += d_diff_a.seconds
            sum_diff_av_1 = sum_diff_1 / DUR_a[y].__len__()
        else:
            sum_diff_av_1 = 0

        if DUR_b[y].__len__() > 0:
            for elem in range(DUR_b[y].__len__()):
                d_start_b = datetime.datetime.strptime(DUR_b[y][elem]["start"], "%Y-%m-%d %H:%M:%S")
                d_end_b = datetime.datetime.strptime(DUR_b[y][elem]["end"], "%Y-%m-%d %H:%M:%S")
                if d_end_b.date() == d_start_b.date():
                    d_diff_b = datetime.timedelta(1)
                else:
                    d_diff_b = d_end_b - d_start_b
                sum_diff_2 += d_diff_b.seconds
                # print(sum_diff_2)
            sum_diff_av_2 = sum_diff_2 / DUR_b[y].__len__()
        else:
            sum_diff_av_2 = 0

        # print(str(sum_diff_av_2) +"|"+ str(sum_diff_av_1))
        if section == "year":
            sum_dev += __calcDev(sum_diff_av_1 / (365 * 24 * 3600),
                                 sum_diff_av_2 / (365 * 24 * 3600))
        elif section == "month":
            sum_dev += __calcDev(sum_diff_av_1 / (30 * 24 * 3600),
                                 sum_diff_av_2 / (30 * 24 * 3600))
        elif section == "day":
            sum_dev += __calcDev(sum_diff_av_1 / (24 * 3600),
                                 sum_diff_av_2 / (24 * 3600))
        elif section == "hour":
            sum_dev += __calcDev(sum_diff_av_1 / (3600), sum_diff_av_2 / (3600))
        else:
            raise HydroSignaturesError("Your section: " + section + " is not valid. See pydoc of this function")

    return sum_dev / DUR_a.__len__()


def __calcFloodDuration(data, datetime_series, threshold_factor, section, which_flow):
    """
    With a given data set we use the datetime_series and save all continuous floods, measured by a given
    threshold_factor times the median of the data. The start and end time of this event is recorded. Based on the user's
    section we create the list of the calculated values per year, month, day, hour.
    Important to know is that the user can input a date-time object with several intervals, so it could be every second 
    or every day recorded data.
    This does not matter at all, we just save the beginning and ending date-time, the difference of threshold and 
    measurement and the amount of how many steps are in the flood event. 
    This function is used by several "getFlood*"-Functions which then calculate the desired hydrological index.

    :param data: measurement / simulation of a flow
    :type data: list
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_factor: which times the median we use for a threshold
    :type threshold_factor: float
    :param section: one of ["year","month","day","hour"]
    :type section: string
    :param which_flow: in ["flood","baseflow"]
    :type which_flow: string
    :return: objects per section with the flood event
    :rtype: dict
    """
    SuitableInput(datetime_series, section)
    duration_per_section = {}
    tmp_duration_logger_per_sec = {}
    threshold_on_per_year = {}

    index = 0

    if which_flow not in ["flood", "baseflow"]:
        raise HydroSignaturesError("which_flow should be flood or baseflow")
    if section not in ["year","month","day","hour"]:
        raise HydroSignaturesError("Your section: " + section + " is not valid. See pydoc of this function")

    tmpStdDurLG = {'start': "0000-00-00", 'end': '0000-00-00', 'duration': 0}
    if which_flow == "flood":
        tmpStdDurLG['overflow'] = []
    elif which_flow == "baseflow":
        tmpStdDurLG['underflow'] = []

    if __isSorted(datetime_series):
        for d in datetime_series:
            if section == "year":
                sec_key = d.to_pydatetime().year
            elif section == "month":
                sec_key = str(d.to_pydatetime().year) + "-" + str(d.to_pydatetime().month)
            elif section == "day":
                sec_key = str(d.to_pydatetime().date())
            elif section == "hour":
                sec_key = str(d.to_pydatetime().date()) + "-" + str(d.to_pydatetime().hour)

            if sec_key not in duration_per_section:
                # Define a bunch of arrays to handle, save and organize the analyze of the data as most as possible at the same time
                duration_per_section[sec_key] = []
                tmp_duration_logger_per_sec[sec_key] = copy.deepcopy(tmpStdDurLG)
                threshold_on_per_year[sec_key] = False

                # And save the old years duration object:
                if index > 0:

                    tmp_lastsec_d = datetime_series[index - 1]
                    if section == "year":
                        tmp_lastsec = tmp_lastsec_d.to_pydatetime().year
                    elif section == "month":
                        tmp_lastsec = str(tmp_lastsec_d.to_pydatetime().year) + "-" + str(
                            tmp_lastsec_d.to_pydatetime().month)
                    elif section == "day":
                        tmp_lastsec = str(tmp_lastsec_d.to_pydatetime().date())
                    elif section == "hour":
                        tmp_lastsec = str(tmp_lastsec_d.to_pydatetime().date()) + "-" + str(
                            tmp_lastsec_d.to_pydatetime().hour)

                    if tmp_duration_logger_per_sec[tmp_lastsec]["duration"] > 0:
                        tmp_duration_logger_per_sec[tmp_lastsec]["end"] = datetime_series[
                            index - 1].to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
                        duration_per_section[tmp_lastsec].append(
                            copy.deepcopy(tmp_duration_logger_per_sec[tmp_lastsec]))
                        tmp_duration_logger_per_sec[tmp_lastsec] = copy.deepcopy(tmpStdDurLG)

            event_happend = False
            if which_flow == "flood":
                if data[index] > threshold_factor * __calcMedianFlow(data):
                    event_happend = True
                    diff = data[index] - threshold_factor * __calcMedianFlow(data)
                    tmp_duration_logger_per_sec[sec_key]["overflow"].append(diff)
                else:
                    event_happend = False
            elif which_flow == "baseflow":
                if data[index] < threshold_factor * __calcMedianFlow(data):
                    event_happend = True
                    diff = data[index] - threshold_factor * __calcMedianFlow(data)
                    tmp_duration_logger_per_sec[sec_key]["underflow"].append(diff)
                else:
                    event_happend = False
            if event_happend:
                if not threshold_on_per_year[sec_key]:
                    threshold_on_per_year[sec_key] = True
                    tmp_duration_logger_per_sec[sec_key]["start"] = d.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
                tmp_duration_logger_per_sec[sec_key]["duration"] = tmp_duration_logger_per_sec[sec_key]["duration"] + 1
            else:
                if threshold_on_per_year[sec_key]:
                    threshold_on_per_year[sec_key] = False
                    tmp_duration_logger_per_sec[sec_key]["end"] = datetime_series[index - 1].to_pydatetime().strftime(
                        "%Y-%m-%d %H:%M:%S")
                    if tmp_duration_logger_per_sec[sec_key]["duration"] > 0:
                        # Here we save the logged flood into the big array
                        duration_per_section[sec_key].append(copy.deepcopy(tmp_duration_logger_per_sec[sec_key]))

                    tmp_duration_logger_per_sec[sec_key] = copy.deepcopy(tmpStdDurLG)
                else:
                    threshold_on_per_year[sec_key] = False
            index += 1
    else:
        raise HydroSignaturesError("The timeseries is not sorted, so a calculation can not be performed")

    return duration_per_section


def getFloodFrequency(evaluation, simulation, datetime_series, threshold_factor=3, section="day"):
    """
    Get high and low-flow event frequencies which have a threshold of "threshold_factor" the median

    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_factor: which times the median we use for a threshold
    :type threshold_factor: float
    :param section: one of ["year","month","day","hour"]
    :type section: string
    :return: mean of deviation of average flood frequency of a defined section
    :rtype: float

    
    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    FRE_s = __calcFlowLevelEventFrequency(simulation, datetime_series, threshold_factor=threshold_factor, section=section, flow_level_type="flood")
    FRE_e = __calcFlowLevelEventFrequency(evaluation, datetime_series, threshold_factor=threshold_factor, section=section, flow_level_type="flood")
    sum = 0.0
    for sec in FRE_s:
        sum += __calcDev(FRE_s[sec], FRE_e[sec])
    return sum / FRE_s.__len__()


def getBaseflowFrequency(evaluation, simulation, datetime_series, threshold_factor=3, section="day"):
    """
    Get high and low-flow event frequencies which have a threshold of "threshold_factor" the median


    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_factor: which times the median we use for a threshold
    :type threshold_factor: float
    :param section: one of ["year","month","day","hour"]
    :type section: string
    :return: mean of deviation of average baseflow frequency of a defined section
    :rtype: float


    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    FRE_s = __calcFlowLevelEventFrequency(simulation, datetime_series, threshold_factor=threshold_factor, section=section, flow_level_type="baseflow")
    FRE_e = __calcFlowLevelEventFrequency(evaluation, datetime_series, threshold_factor=threshold_factor, section=section, flow_level_type="baseflow")
    sum = 0.0
    for sec in FRE_s:
        sum += __calcDev(FRE_s[sec], FRE_e[sec])
    return sum / FRE_s.__len__()


def __calcFlowLevelEventFrequency(data, datetime_series, threshold_factor, section, flow_level_type):
    """
        Calc the high and low-flow event frequencies which have a threshold of "threshold_factor" the median
        
        :param data: data where the flood frequency is calculated of
        :type data: list
        :param datetime_series: a pandas data object with sorted (may not be complete but sorted) dates
        :type datetime_series: pandas datetime
        :param threshold_factor: which times the median as threshold calculation should be used.
        :type threshold_factor: float
        :param section: for which section should the function calc a frequency of flood
        :type section: string
        :param flow_level_type: in ["flood","baseflow"]:
        :type flow_level_type: string
        :return: mean of deviation of average frequency of a defined section, allowed is ["year","month","day","hour"]
        :rtype: float
    """

    if flow_level_type not in ["flood","baseflow"]:
        raise HydroSignaturesError("flow_level_type should flood or baseflow")

    if __isSorted(datetime_series):

        count_per_section = {}
        index = 0

        for d in datetime_series:
            if section == "year":
                sec_key = d.to_pydatetime().year
            elif section == "month":
                sec_key = str(d.to_pydatetime().year) + "-" + str(d.to_pydatetime().month)
            elif section == "day":
                sec_key = str(d.to_pydatetime().date())
            elif section == "hour":
                sec_key = str(d.to_pydatetime().date()) + "-" + str(d.to_pydatetime().hour)
            else:
                raise HydroSignaturesError("Your section: " + section + " is not valid. See pydoc of this function")

            if sec_key not in count_per_section:
                count_per_section[sec_key] = 0

            if flow_level_type == "flood":
                if data[index] > threshold_factor * __calcMedianFlow(data):
                    count_per_section[sec_key] += 1
            elif flow_level_type == "baseflow":
                if data[index] < threshold_factor * __calcMedianFlow(data):
                    count_per_section[sec_key] += 1
            index += 1

    else:
        raise HydroSignaturesError("The time series is not sorted, so a calculation can not be performed")

    return count_per_section


def getLowFlowVar(evaluation, simulation, datetime_series):
    """
    
    Mean of annual minimum flow divided by the median flow (Jowett and Duncan, 1990)
     
    Annular Data
    
        .. math::
        
         Annualar Data= \\frac{\\sum_{i=1}^{N}(min(d_i)}{N*median(data)}


    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    :param datetime_series: a pandas data object with sorted (may not be complete but sorted) dates
    :type datetime_series: pandas datetime object
    :return: mean of deviation of the low flow variation
    :rtype: float

    """

    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")
    if not __isSorted(datetime_series):
        raise HydroSignaturesError("datetime_series data are not sorted")

    sim_LFV = __calcAnnularData(simulation, datetime_series, "min") / __calcMedianFlow(simulation)
    obs_LFV = __calcAnnularData(evaluation, datetime_series, "min") / __calcMedianFlow(evaluation)
    return __calcDev(sim_LFV, obs_LFV)


def getHighFlowVar(evaluation, simulation, datetime_series):
    """
    Mean of annual maximum flow divided by the median flow (Jowett and Duncan, 1990)

    Annular Data
    
        .. math::
        
         Annualar Data= \\frac{\\sum_{i=1}^{N}(max(d_i)}{N*median(data)}



    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    :param datetime_series: a pandas data object with sorted (may not be complete but sorted) dates
    :type datetime_series: pandas datetime object
    :return: mean of deviation of the high flow variation
    :rtype: float

    """

    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")
    if not __isSorted(datetime_series):
        raise HydroSignaturesError("datetime_series data are not sorted")

    sim_LFV = __calcAnnularData(simulation, datetime_series, "max") / __calcMedianFlow(simulation)
    obs_LFV = __calcAnnularData(evaluation, datetime_series, "max") / __calcMedianFlow(evaluation)
    return __calcDev(sim_LFV, obs_LFV)


def __calcAnnularData(data, datetime_series, what):
    """
    Annular Data
    
    :math:`Annualar Data= \\frac{\\sum_{i=1}^{N}(max(d_i)}{N}`

    :param data: measurements
    :type data: list
    :param datetime_series: sorted pandas date time object
    :type datetime_series: pandas datetime object

    :param what: string which switches the calculation method. Allowed are: min (the min value of a year) and max (the max value of a year)
    :type what: string

    :return:mean of min/max per year
    :rtype: float
    """
    data_per_year_tmp = []
    data_per_year = {}
    index = 0
    for d in datetime_series:
        yr = d.to_pydatetime().year
        if yr not in data_per_year:
            if index > 0:
                data_per_year[datetime_series[index - 1].to_pydatetime().year] = data_per_year_tmp
                data_per_year[datetime_series[index - 1].to_pydatetime().year] = data_per_year_tmp
                data_per_year_tmp = []
        data_per_year_tmp.append(data[index])
        index += 1

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
        raise HydroSignaturesError("The parameter what=" + what + " is not defined")


def getBaseflowIndex(evaluation, simulation, datetime_series):
    """
    We may have to use baseflow devided with total discharge
    See https://de.wikipedia.org/wiki/Niedrigwasser and
    see also http://people.ucalgary.ca/~hayashi/kumamoto_2014/lectures/2_3_baseflow.pdf

    For the formular look at: IH_108.pdf
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    
    :param datetime_series: a pandas data object with sorted (may not be complete but sorted) dates
    :type datetime_series: pandas datetime
    
    :return: deviation of base flow index
    :rtype: float
    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    if simulation.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")
    if not __isSorted(datetime_series):
        raise HydroSignaturesError("datetime_series data are not sorted")

    bfi_sim = __calcBaseflowIndex(simulation, datetime_series)
    bfi_obs = __calcBaseflowIndex(evaluation, datetime_series)
    sum_sim = 0.0
    sum_obs = 0.0

    for y in bfi_obs:
        sum_obs += bfi_obs[y]
    for y in bfi_sim:
        sum_sim += bfi_sim[y]

    sum_obs = sum_obs / bfi_obs.__len__()
    sum_sim = sum_sim / bfi_sim.__len__()

    return __calcDev(sum_sim, sum_obs)


def __calcBaseflowIndex(data, datetime_series):
    """
    Basefow Index

    :math:`BasefowIndex = \\frac{BF}{TD}` where BF is the median of the data and TD the minimum of the data per year

    :param data: float list
    :type data: list
    :param datetime_series: sorted pandas datetime object
    :type datetime_series: pandas datetime object
    :return: BFI per year
    :rtype: dict
    """
    Min_per_year = {}
    Q50_per_year = {}
    data_per_year_tmp = []
    index = 0
    for d in datetime_series:
        yr = d.to_pydatetime().year
        if yr not in Min_per_year:
            if index > 0:
                Min_per_year[datetime_series[index - 1].to_pydatetime().year] = np.min(data_per_year_tmp)
                Q50_per_year[datetime_series[index - 1].to_pydatetime().year] = np.median(data_per_year_tmp)
                data_per_year_tmp = []
        data_per_year_tmp.append(data[index])
        index += 1

    BFI = {}
    for y in Min_per_year:
        BFI[y] = Q50_per_year[y] / Min_per_year[y]
    return BFI


def getSlopeFDC(evaluation, simulation):
    """
    Slope of the FDC between the 33 and 66 % exceedance values of streamflow normalised by its mean (Yadav et al., 2007)
     
    
    :param evaluation: Observed data to compared with simulation data.
    :type evaluation: list
    
    :param simulation: simulation data to compared with evaluation data
    :type simulation: list
    
    :return: deviation of the slope
    :rtype: float
    """
    if simulation.__len__() != evaluation.__len__():
        raise HydroSignaturesError("Simulation and observation data have not the same length")

    return __calcDev(__calcSlopeFDC(simulation), __calcSlopeFDC(evaluation))


def __calcSlopeFDC(data):
    """
    The main idea is to use a threshold by the mean of the data and use the first occurrence of a 33% exceed and a 66% 
    exceed and calculate the factor of how many times is the 66% exceed higher then the 33% exceed.
    If 33% or 66% exceed does not exists then just give 0 back for a slope of 0 (horizontal line) 
    
    :math:`slope = \\frac{treshold(mean*1,33 <= data)}{treshold(mean*1,66 <= data)}`
    
    :param data: float list
    :type data: list
    :return: float slope
    :rtype: float
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
