# -*- coding: utf-8 -*-
'''
Copyright 2017 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska, Benjamin Manns
This module contains a framework to analyze data sets of hydrological signatures.
'''


import numpy as np
import copy
import warnings
import datetime

try:
    import pandas
except ImportError:
    print('Please install Pandas to use these signature functions')
    
   

class SuitableInput:
    def __init__(self, datm):
        """
        Calculates which section type the the date time series suits best (year, month, day, hour).

        :param datm:
        :type datm: pandas datetime object
        :return: the section type which suites best
        :rtype: string
        """
        self.datm = datm
        self.allowed_sections = ['year', 'month', 'day', 'hour']

    def calc(self):
        if self.datm.__len__() > 1:
            diff = (self.datm[1].to_pydatetime() - self.datm[0].to_pydatetime()).total_seconds()
            anything_found = False
            found_section = ''
            while not anything_found:
                try:
                    section = self.allowed_sections.pop()
                except IndexError:
                    break

                if section == "year":
                    anything_found, found_section = diff / (3600 * 24 * 365) <= 1.0, 'year'
                elif section == "month":
                    anything_found, found_section = diff / (3600 * 24 * 30) <= 1.0, 'month'
                elif section == "day":
                    anything_found, found_section = diff / (3600 * 24) <= 1.0, 'day'
                elif section == "hour":
                    anything_found, found_section = diff <= 3600, 'hour'
                else:
                    raise Exception("The section [" + section + "] is not defined in " + str(self))
            return found_section


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


def __calcMedianFlow(data):
    """
    Simply calculate the median (flow exceeded 50% of the time) of the data

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: A list of float data
    :type data: list
    :return: Median
    :rtype: float
    """
    return np.percentile(data, 50)


def __percentilwrapper(array, index):
    """
    A Percentil Wrapper to have a easy chance to modify the percentil calculation for the following functions

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

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

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: A list of float data
    :type data: list
    :return: Mean
    :rtype: float
    """
    return np.mean(data)


class HydroSignaturesError(Exception):
    """
    Define an own error class to know it is an error made by a hydroindex calculation to warn the use for wrong inputs
    """
    pass


class _SignaturesBasicFunctionality(object):
    def __init__(self, data, comparedata=None, mode=None):
        """
        A basic class to give a blueprint of a signature calculation class
        :param data: data to analyze
        :type data: list
        :param comparedata: data to analyze and compare with variable data
        :type comparedata: list
        :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
        :type mode: string
        """
        self.data = data
        self.mode = mode
        self.additional_arguments = {}
        self.preProcessFunction_additional_arguments = {}
        self.hasPreProcess = False
        self.preProcessFunction = None

        if comparedata is not None:
            self.comparedata = comparedata
            if data.__len__() != comparedata.__len__():
                raise HydroSignaturesError("Simulation and observation data have not the same length")
            self.which_case = "compare"
        else:
            self.which_case = "single"

    def pre_process(self, pre_func=None, **kwargs):
        if pre_func is not None:
            self.hasPreProcess = True
            for k, v in kwargs.items():
                self.preProcessFunction_additional_arguments[k] = v
        self.preProcessFunction = pre_func

    def analyze(self, func, **kwargs):

        # pack the kwargs into a dict so it can be expressed as keyword arguments later on

        for k, v in kwargs.items():
            self.additional_arguments[k] = v

        if self.mode == 'get_signature':
            #print('Calculation Signature')
            return self.get_signature(func)
        elif self.mode == 'get_raw_data':
            #print('Calculation raw data')
            return self.raw_data(func)
        elif self.mode == 'calc_Dev':
            #print('Calculation difference')
            try:
                return self.run_compared_signature(func)
            except AttributeError:
                warnings.warn("The signature method need a not None value for `comparedata` if you use `calc_Dev`")
                return None

        else:
            raise ValueError(
                "'%s' is not a valid keyword for signature processing" % self.mode)

    def get_signature(self, func):
        if self.which_case == "compare":
            return self.run_compared_signature(func)
        elif self.which_case == "single":
            return self.run_signature(self.data, func)
        else:
            raise HydroSignaturesError("No case set!")

    def raw_data(self, func):
        if func.__name__ == "__calcFloodDuration":
            return func(self.data, **self.additional_arguments)[1]
        else:
            return func(self.data, **self.additional_arguments)

    def run_signature(self, data, func):
        return func(data, **self.additional_arguments)

    def run_compared_signature(self, func):
        if not self.hasPreProcess:
            return self.calcDev(self.run_signature(self.data, func), self.run_signature(self.comparedata, func))
        else:
            return self.preProcessFunction(self.run_signature(self.data, func),
                                           self.run_signature(self.comparedata, func),
                                           **self.preProcessFunction_additional_arguments)

    @staticmethod
    def calcDev(a, b):
        """
        Calculate the relative error / derivation of two values
        If one parameter is zero the result is just 1, for example b = 0, so calculate:
        :math:`\\frac{a+0}{a} = 1` and also
        :math:`a =  b  \\Leftrightarrow  return =  0` [approximately]

        See https://en.wikipedia.org/wiki/Approximation_error

        :param a: Value a
        :type a: float
        :param b: Value b
        :type b: float
        :return: relative error of a and b (numeric definition)
        :rtype: float
        """
        if type(a) == type({}) or type(b) == type({}) or a is None or b is None:
            raise HydroSignaturesError("You specified no pre process because your data are in a dict or NONE!")
        else:

            if a != 0:
                return (a - b) / a
            elif b != 0:
                return (a - b) / b
            else:
                return 0


def getSlopeFDC(data, comparedata=None, mode='get_signature'):
    """
    The main idea is to use a threshold by the mean of the data and use the first occurrence of a 33% exceed and a 66%
    exceed and calculate the factor of how many times is the 66% exceed higher then the 33% exceed.
    If 33% or 66% exceed does not exists then just give 0 back for a slope of 0 (horizontal line)

    :math:`slope = \\frac{treshold(mean*1,33 <= data)}{treshold(mean*1,66 <= data)}`

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string
    :return: the calculation will be return which is set by mode, this might be float numbers or war data in dict format which can be plot to visualize the signatures
    :rtype: dict / float

    """
    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)

    return basics.analyze(__calcSlopeFDC)


def __calcSlopeFDC(data):
    upper33_data = np.sort(data)[np.sort(data) >= 1.33 * np.mean(data)]
    upper66_data = np.sort(data)[np.sort(data) >= 1.66 * np.mean(data)]
    if upper33_data.__len__() > 0 and upper66_data.__len__() > 0:
        if upper66_data[0] != 0:
            return upper33_data[0] / upper66_data[0]
        else:
            return 0.0
    else:
        return 0.0


def getAverageFloodOverflowPerSection(data, comparedata=None, mode='get_signature', datetime_series=None,
                                      threshold_value=3):
    """
    All measurements are scanned where there are overflow events. Based on the section we summarize events per year,
    month, day, hour.
    Of course we need a datetime_series which has the the suitable resolution. So, for example, if you group the
    overflow events hourly but you have only daily data the function will work but not very useful.

    However for every section the function collect the overflow value, i.e. value - threshold and calc the deviation
    of the means of this overflows.

    The idea is based on values from "REDUNDANCY AND THE CHOICE OF HYDROLOGIC INDICES FOR CHARACTERIZING STREAMFLOW REGIMES
    JULIAN D. OLDEN* and N. L. POFF" (RiverResearchApp_2003.pdf, page 109). An algorithms to calculate this data is not
    given, so we developed an own.

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_value: which times the median we use for a threshold
    :type threshold_value: float
    :return: deviation of means of overflow value or raw data
    :rtype: dict / float
    """

    if datetime_series is None:
        raise HydroSignaturesError("datetime_series is None. Please specify a datetime_series.")

    if data.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)

    basics.pre_process(__calcDeviationForAverageFloodOverflowPerSection)
    return basics.analyze(__calcFloodDuration, datetime_series=datetime_series, threshold_value=threshold_value,
                          which_flow="flood")


def __calcDeviationForAverageFloodOverflowPerSection(a, b):
    for_mean_a = []
    for_mean_b = []
    a = a[0]
    b = b[0]

    for y in a:
        if a[y].__len__() > 0:
            for elem in range(a[y].__len__()):
                for ov in a[y][elem]["overflow"]:
                    for_mean_a.append(ov)
        for y in b:
            if b[y].__len__() > 0:
                for elem in range(b[y].__len__()):
                    for ov in b[y][elem]["overflow"]:
                        for_mean_b.append(ov)
    if for_mean_a.__len__() > 0 and for_mean_b.__len__() > 0:
        return _SignaturesBasicFunctionality.calcDev(np.mean(for_mean_a), np.mean(for_mean_b))
    else:
        warnings.warn("No data in for_mean_a to calculate mean of")


def __calcFloodDuration(data, datetime_series, threshold_value, which_flow):
    """
    With a given data set we use the datetime_series and save all continuous floods, measured by a given
    threshold_value times the median of the data. The start and end time of this event is recorded. Based on the best
    suitable section we create the list of the calculated values per year, month, day, hour.
    Important to know is that the user can input a date-time object with several intervals, so it could be every second
    or every day recorded data.
    This does not matter at all, we just save the beginning and ending date-time, the difference of threshold and
    measurement and the amount of how many steps are in the flood event.
    This function is used by several "getFlood*"-Functions which then calculate the desired hydrological index.

    The idea is based on values from "REDUNDANCY AND THE CHOICE OF HYDROLOGIC INDICES FOR CHARACTERIZING STREAMFLOW REGIMES
    JULIAN D. OLDEN* and N. L. POFF" (RiverResearchApp_2003.pdf, page 109). An algorithms to calculate this data is not
    given, so we developed an own.

    :param data: measurement / simulation of a flow
    :type data: list
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_value: which times the median we use for a threshold
    :type threshold_value: float
    :param which_flow: in ["flood","baseflow"]
    :type which_flow: string
    :return: objects per section with the flood event
    :rtype: dict
    """
    s = SuitableInput(datetime_series)
    section = s.calc()
    duration_per_section = {}
    tmp_duration_logger_per_sec = {}
    threshold_on_per_year = {}
    event_mapped_to_datetime = []
    index = 0

    if which_flow not in ["flood", "baseflow"]:
        raise HydroSignaturesError("which_flow should be flood or baseflow")
    if section not in ["year", "month", "day", "hour"]:
        raise HydroSignaturesError("Your section: " + section + " is not valid. See pydoc of this function")

    tmpStdDurLG = {'start': "0000-00-00", 'end': '0000-00-00', 'duration': 0,}
    if which_flow == "flood":
        tmpStdDurLG['overflow'] = []
    elif which_flow == "baseflow":
        tmpStdDurLG['underflow'] = []

    if __isSorted(datetime_series):
        for d in datetime_series:
            if section == "year":
                sec_key = pandas.Timestamp(datetime.datetime(year=d.to_pydatetime().year, month=1, day=1))
            elif section == "month":
                sec_key = pandas.Timestamp(
                    datetime.datetime(year=d.to_pydatetime().year, month=d.to_pydatetime().month, day=1))
            elif section == "day":
                sec_key = d
            elif section == "hour":
                sec_key = pandas.Timestamp(datetime.datetime(year=d.to_pydatetime().year, month=d.to_pydatetime().month,
                                                             day=d.to_pydatetime().day, hour=d.to_pydatetime().hour))
            else:
                raise HydroSignaturesError("Your section: " + section + " is not valid. See pydoc of this function")

            if sec_key not in duration_per_section:
                # Define a bunch of arrays to handle, save and organize the analyze of the data as most as possible at the same time
                duration_per_section[sec_key] = []
                tmp_duration_logger_per_sec[sec_key] = copy.deepcopy(tmpStdDurLG)
                threshold_on_per_year[sec_key] = False

                # And save the old years duration object:
                if index > 0:

                    tmp_lastsec_d = datetime_series[index - 1]
                    if section == "year":
                        tmp_lastsec = pandas.Timestamp(datetime.datetime(year=tmp_lastsec_d.to_pydatetime().year, month=1, day=1))
                    elif section == "month":
                        tmp_lastsec = pandas.Timestamp(
                            datetime.datetime(year=tmp_lastsec_d.to_pydatetime().year, month=tmp_lastsec_d.to_pydatetime().month, day=1))
                    elif section == "day":
                        tmp_lastsec = tmp_lastsec_d
                    elif section == "hour":
                        tmp_lastsec = pandas.Timestamp(
                            datetime.datetime(year=tmp_lastsec_d.to_pydatetime().year, month=tmp_lastsec_d.to_pydatetime().month,
                                              day=tmp_lastsec_d.to_pydatetime().day, hour=tmp_lastsec_d.to_pydatetime().hour))
                    else:
                        raise HydroSignaturesError(
                            "Your section: " + section + " is not valid. See pydoc of this function")


                    if tmp_duration_logger_per_sec[tmp_lastsec]["duration"] > 0:
                        tmp_duration_logger_per_sec[tmp_lastsec]["end"] = datetime_series[
                            index - 1].to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
                        duration_per_section[tmp_lastsec].append(
                            copy.deepcopy(tmp_duration_logger_per_sec[tmp_lastsec]))
                        tmp_duration_logger_per_sec[tmp_lastsec] = copy.deepcopy(tmpStdDurLG)

            event_happend = False
            if which_flow == "flood":
                if data[index] > threshold_value:
                    event_happend = True
                    diff = data[index] - threshold_value
                    tmp_duration_logger_per_sec[sec_key]["overflow"].append(diff)
                else:
                    event_happend = False
            elif which_flow == "baseflow":
                if data[index] < threshold_value:
                    event_happend = True
                    diff = data[index] - threshold_value
                    tmp_duration_logger_per_sec[sec_key]["underflow"].append(diff)
                else:
                    event_happend = False

            tmp_dict_for_eventMap = {"datetime": d}
            if event_happend:


                tmp_dict_for_eventMap[which_flow]=diff
                event_mapped_to_datetime.append(tmp_dict_for_eventMap)
                if not threshold_on_per_year[sec_key]:
                    threshold_on_per_year[sec_key] = True
                    tmp_duration_logger_per_sec[sec_key]["start"] = d.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
                tmp_duration_logger_per_sec[sec_key]["duration"] = tmp_duration_logger_per_sec[sec_key]["duration"] + 1
            else:
                tmp_dict_for_eventMap[which_flow] = 0
                event_mapped_to_datetime.append(tmp_dict_for_eventMap)
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

    event_mapped_to_datetime = pandas.DataFrame(event_mapped_to_datetime)
    event_mapped_to_datetime = event_mapped_to_datetime.set_index("datetime")
    return duration_per_section, event_mapped_to_datetime


def getMeanFlow(data, comparedata=None, mode='get_signature'):
    """
    Simply calculate the mean of the data

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: Mean
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__calcMeanFlow)


def getMedianFlow(data, comparedata=None, mode='get_signature'):
    """
    Simply calculate the median (flow exceeded 50% of the time) of the data

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: Median
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__calcMedianFlow)


def getSkewness(data, comparedata=None, mode='get_signature'):
    """
    Skewness, i.e. the mean flow data divided by Q50 (50 percentil / median flow) .

    See paper "B. Clausen, B.J.F. Biggs / Journal of Hydrology 237 (2000) 184-197", (M_A1_MeanDailyFlows .pdf,  page 185)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: derivation of the skewness
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__Skewness)


def __Skewness(data):
    return __calcMeanFlow(data) / __calcMedianFlow(data)


def getCoeffVariation(data, comparedata=None, mode='get_signature'):
    """
    Coefficient of variation, i.e. standard deviation divided by mean flow

    See paper "B. Clausen, B.J.F. Biggs / Journal of Hydrology 237 (2000) 184-197", (M_A1_MeanDailyFlows .pdf,  page 185)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: derivation of the coefficient of variation
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(_CoeffVariation)


def _CoeffVariation(data):
    return np.std(data) / __calcMeanFlow(data)


def getQ001(data, comparedata=None, mode='get_signature'):
    """
    The value of the 0.01 percentiles

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: derivation of the 0.01 percentiles
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__percentilwrapper, index=0.01)


def getQ01(data, comparedata=None, mode='get_signature'):
    """
    The value of the 0.1 percentiles

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: derivation of the 0.1 percentiles
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__percentilwrapper, index=0.1)


def getQ1(data, comparedata=None, mode='get_signature'):
    """
    The value of the 1 percentiles

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: derivation of the 1 percentiles
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__percentilwrapper, index=1)


def getQ5(data, comparedata=None, mode='get_signature'):
    """
    The value of the 5 percentiles

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: derivation of the 5 percentiles
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__percentilwrapper, index=1)


def getQ10(data, comparedata=None, mode='get_signature'):
    """
    The value of the 10 percentiles

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: derivation of the 10 percentiles
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__percentilwrapper, index=10)


def getQ20(data, comparedata=None, mode='get_signature'):
    """
    The value of the 20 percentiles

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: derivation of the 20 percentiles
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__percentilwrapper, index=20)


def getQ85(data, comparedata=None, mode='get_signature'):
    """
    The value of the 85 percentiles

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: derivation of the 85 percentiles
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__percentilwrapper, index=85)


def getQ95(data, comparedata=None, mode='get_signature'):
    """
    The value of the 95 percentiles

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: derivation of the 95 percentiles
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__percentilwrapper, index=95)


def getQ99(data, comparedata=None, mode='get_signature'):
    """
    The value of the 99 percentiles

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :return: derivation of the 99 percentiles
    :rtype: float

    """
    basics = _SignaturesBasicFunctionality(data=data, comparedata=comparedata, mode=mode)
    return basics.analyze(__percentilwrapper, index=99)


def getAverageFloodFrequencyPerSection(data, comparedata=None, mode='get_signature', datetime_series=None,
                                       threshold_value=3):
    """
    This function calculates the average frequency per every section in the given interval of the datetime_series.
    So if the datetime is recorded all 5 min we use this fine interval to count all records which are in flood.

    The idea is based on values from "REDUNDANCY AND THE CHOICE OF HYDROLOGIC INDICES FOR CHARACTERIZING STREAMFLOW REGIMES
    JULIAN D. OLDEN* and N. L. POFF" (RiverResearchApp_2003.pdf, page 109). An algorithms to calculate this data is not
    given, so we developed an own.

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_value: a threshold where a flood event is defined with
    :type threshold_value: float
    :return: deviation of means of flood frequency per best suitable section or a whole dict if mode was set to raw data
    :rtype: float / dict
    """

    if datetime_series is None:
        raise HydroSignaturesError("datetime_series is None. Please specify a datetime_series.")

    if data.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)

    basics.pre_process(__calcDeviationForAverageFloodFrequencyPerSection)
    return basics.analyze(__calcFloodDuration, datetime_series=datetime_series, threshold_value=threshold_value,
                          which_flow="flood")


def __calcDeviationForAverageFloodFrequencyPerSection(a, b):
    sum_dev = 0.0
    a = a[0]
    b = b[0]
    for y in a:
        sum_dur_1 = 0.0
        sum_dur_2 = 0.0
        for elem in a[y]:
            sum_dur_1 += elem["duration"]
        for elem in b[y]:
            sum_dur_2 += elem["duration"]

        sum_dev += _SignaturesBasicFunctionality.calcDev(sum_dur_1, sum_dur_2)
    return sum_dev / a.__len__()


def getAverageFloodDuration(data, comparedata=None, mode='get_signature', datetime_series=None, threshold_value=3):
    """
    Get high and low-flow yearly-average event duration which has a threshold of threshold_value, this could be any float

    The idea is based on values from "REDUNDANCY AND THE CHOICE OF HYDROLOGIC INDICES FOR CHARACTERIZING STREAMFLOW REGIMES
    JULIAN D. OLDEN* and N. L. POFF" (RiverResearchApp_2003.pdf, page 109). An algorithms to calculate this data is not
    given, so we developed an own.

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_value: a threshold where a flood event is defined with
    :type threshold_value: float
    :return: deviation of means of flood durations or a dict if mode was set to raw data
    :rtype: float / dict
    """

    if datetime_series is None:
        raise HydroSignaturesError("datetime_series is None. Please specify a datetime_series.")

    if data.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)

    s = SuitableInput(datetime_series)
    section = s.calc()
    basics.pre_process(__calcDeviationForAverageFloodDuration, section=section)
    return basics.analyze(__calcFloodDuration, datetime_series=datetime_series, threshold_value=threshold_value,
                          which_flow="flood")


def __calcDeviationForAverageFloodDuration(a, b, section):
    sum_dev = 0.0
    a = a[0]

    b = b[0]
    for y in a:
        sum_diff_1 = 0.0
        sum_diff_2 = 0.0
        if a[y].__len__() > 0:
            for elem in range(a[y].__len__()):
                d_start_a = datetime.datetime.strptime(a[y][elem]["start"], "%Y-%m-%d %H:%M:%S")
                d_end_a = datetime.datetime.strptime(a[y][elem]["end"], "%Y-%m-%d %H:%M:%S")
                if d_end_a.date() == d_start_a.date():
                    sum_diff_1 += 24 * 3600
                else:
                    d_diff_a = d_end_a - d_start_a
                    sum_diff_1 += d_diff_a.seconds
            sum_diff_av_1 = sum_diff_1 / a[y].__len__()
        else:
            sum_diff_av_1 = 0

        if b[y].__len__() > 0:
            for elem in range(b[y].__len__()):
                d_start_b = datetime.datetime.strptime(b[y][elem]["start"], "%Y-%m-%d %H:%M:%S")
                d_end_b = datetime.datetime.strptime(b[y][elem]["end"], "%Y-%m-%d %H:%M:%S")
                if d_end_b.date() == d_start_b.date():
                    d_diff_b = datetime.timedelta(1)
                else:
                    d_diff_b = d_end_b - d_start_b
                sum_diff_2 += d_diff_b.seconds
            sum_diff_av_2 = sum_diff_2 / b[y].__len__()
        else:
            sum_diff_av_2 = 0

        if section == "year":
            sum_dev += _SignaturesBasicFunctionality.calcDev(sum_diff_av_1 / (365 * 24 * 3600),
                                                             sum_diff_av_2 / (365 * 24 * 3600))
        elif section == "month":
            sum_dev += _SignaturesBasicFunctionality.calcDev(sum_diff_av_1 / (30 * 24 * 3600),
                                                             sum_diff_av_2 / (30 * 24 * 3600))
        elif section == "day":
            sum_dev += _SignaturesBasicFunctionality.calcDev(sum_diff_av_1 / (24 * 3600),
                                                             sum_diff_av_2 / (24 * 3600))
        elif section == "hour":
            sum_dev += _SignaturesBasicFunctionality.calcDev(sum_diff_av_1 / (3600), sum_diff_av_2 / (3600))
        else:
            raise HydroSignaturesError("Your section: " + section + " is not valid. See pydoc of this function")

    return sum_dev / a.__len__()


def getAverageBaseflowUnderflowPerSection(data, comparedata=None, mode='get_signature', datetime_series=None,
                                          threshold_value=3):
    """
    All measurements are scanned where there are overflow events. Based on the best suitable section we summarize events
    per year, month, day, hour.

    However for every section the function collect the overflow value, i.e. value - threshold  and calc the deviation
    of the means of this overflows.

    The idea is based on values from "REDUNDANCY AND THE CHOICE OF HYDROLOGIC INDICES FOR CHARACTERIZING STREAMFLOW REGIMES
    JULIAN D. OLDEN* and N. L. POFF" (RiverResearchApp_2003.pdf, page 109). An algorithms to calculate this data is not
    given, so we developed an own.

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_value: a threshold where a baseflow event is defined with
    :type threshold_value: float
    :return: deviation of means of underflow value or a dict if mode was set to raw data
    :rtype: float / dict

    """

    if datetime_series is None:
        raise HydroSignaturesError("datetime_series is None. Please specify a datetime_series.")

    if data.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)

    basics.pre_process(__calcDeviationForAverageBaseflowUnderflowPerSection)
    return basics.analyze(__calcFloodDuration, datetime_series=datetime_series, threshold_value=threshold_value,
                          which_flow="baseflow")


def __calcDeviationForAverageBaseflowUnderflowPerSection(a, b):
    for_mean_a = []
    for_mean_b = []
    a = a[0]
    b = b[0]
    for y in a:
        if a[y].__len__() > 0:
            for elem in range(a[y].__len__()):
                for ov in a[y][elem]["underflow"]:
                    for_mean_a.append(ov)
        for y in b:
            if b[y].__len__() > 0:
                for elem in range(b[y].__len__()):
                    for ov in b[y][elem]["underflow"]:
                        for_mean_b.append(ov)

    return _SignaturesBasicFunctionality.calcDev(np.mean(for_mean_a), np.mean(for_mean_b))


def getAverageBaseflowFrequencyPerSection(data, comparedata=None, mode='get_signature', datetime_series=None,
                                          threshold_value=3):
    """
    This function calculates the average frequency per every section in the given interval of the datetime_series.
    So if the datetime is recorded all 5 min we use this fine interval to count all records which are in flood.

    The idea is based on values from "REDUNDANCY AND THE CHOICE OF HYDROLOGIC INDICES FOR CHARACTERIZING STREAMFLOW REGIMES
    JULIAN D. OLDEN* and N. L. POFF" (RiverResearchApp_2003.pdf, page 109). An algorithms to calculate this data is not
    given, so we developed an own.

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_value: a threshold where a baseflow event is defined with
    :type threshold_value: float
    :return: deviation of means of baseflow frequency per section or a dict if mode was set to raw data
    :rtype: float / dict
    """

    if datetime_series is None:
        raise HydroSignaturesError("datetime_series is None. Please specify a datetime_series.")

    if data.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)

    basics.pre_process(__calcDevForAverageBaseflowFrequencyPerSection)
    return basics.analyze(__calcFloodDuration, datetime_series=datetime_series, threshold_value=threshold_value,
                          which_flow="baseflow")


def __calcDevForAverageBaseflowFrequencyPerSection(a, b):
    sum_dev = 0.0
    a = a[0]
    b = b[0]
    for y in a:
        sum_dur_1 = 0.0
        sum_dur_2 = 0.0
        for elem in a[y]:
            sum_dur_1 += elem["duration"]
        for elem in b[y]:
            sum_dur_2 += elem["duration"]

        sum_dev += _SignaturesBasicFunctionality.calcDev(sum_dur_1, sum_dur_2)

    return sum_dev / a.__len__()


def getAverageBaseflowDuration(data, comparedata=None, mode='get_signature', datetime_series=None, threshold_value=3):
    """
    Get high and low-flow yearly-average event duration which have a threshold of threshold_value

    The idea is based on values from "REDUNDANCY AND THE CHOICE OF HYDROLOGIC INDICES FOR CHARACTERIZING STREAMFLOW REGIMES
    JULIAN D. OLDEN* and N. L. POFF" (RiverResearchApp_2003.pdf, page 109). An algorithms to calculate this data is not
    given, so we developed an own.

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_value: a threshold where a baseflow event is defined with
    :type threshold_value: float
    :return: deviation of means of baseflow duration or a dict if mode was set to raw data
    :rtype: float / dict

    """

    if datetime_series is None:
        raise HydroSignaturesError("datetime_series is None. Please specify a datetime_series.")

    if data.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)

    s = SuitableInput(datetime_series)
    section = s.calc()
    basics.pre_process(__calcDevForAverageBaseflowDuration, section=section)
    return basics.analyze(__calcFloodDuration, datetime_series=datetime_series, threshold_value=threshold_value,
                          which_flow="baseflow")


def __calcDevForAverageBaseflowDuration(a, b, section):
    sum_dev = 0.0
    a = a[0]
    b = b[0]

    for y in a:
        sum_diff_1 = 0.0
        sum_diff_2 = 0.0
        if a[y].__len__() > 0:
            for elem in range(a[y].__len__()):
                d_start_a = datetime.datetime.strptime(a[y][elem]["start"], "%Y-%m-%d %H:%M:%S")
                d_end_a = datetime.datetime.strptime(a[y][elem]["end"], "%Y-%m-%d %H:%M:%S")
                if d_end_a.date() == d_start_a.date():
                    sum_diff_1 += 24 * 3600
                else:
                    d_diff_a = d_end_a - d_start_a
                    sum_diff_1 += d_diff_a.seconds
            sum_diff_av_1 = sum_diff_1 / a[y].__len__()
        else:
            sum_diff_av_1 = 0

        if b[y].__len__() > 0:
            for elem in range(b[y].__len__()):
                d_start_b = datetime.datetime.strptime(b[y][elem]["start"], "%Y-%m-%d %H:%M:%S")
                d_end_b = datetime.datetime.strptime(b[y][elem]["end"], "%Y-%m-%d %H:%M:%S")
                if d_end_b.date() == d_start_b.date():
                    d_diff_b = datetime.timedelta(1)
                else:
                    d_diff_b = d_end_b - d_start_b
                sum_diff_2 += d_diff_b.seconds
            sum_diff_av_2 = sum_diff_2 / b[y].__len__()
        else:
            sum_diff_av_2 = 0

        if section == "year":
            sum_dev += _SignaturesBasicFunctionality.calcDev(sum_diff_av_1 / (365 * 24 * 3600),
                                                             sum_diff_av_2 / (365 * 24 * 3600))
        elif section == "month":
            sum_dev += _SignaturesBasicFunctionality.calcDev(sum_diff_av_1 / (30 * 24 * 3600),
                                                             sum_diff_av_2 / (30 * 24 * 3600))
        elif section == "day":
            sum_dev += _SignaturesBasicFunctionality.calcDev(sum_diff_av_1 / (24 * 3600),
                                                             sum_diff_av_2 / (24 * 3600))
        elif section == "hour":
            sum_dev += _SignaturesBasicFunctionality.calcDev(sum_diff_av_1 / (3600), sum_diff_av_2 / (3600))
        else:
            raise HydroSignaturesError("Your section: " + section + " is not valid. See pydoc of this function")

    return sum_dev / a.__len__()


def getFloodFrequency(data, comparedata=None, mode='get_signature', datetime_series=None, threshold_value=3):
    """
    Get high and low-flow event frequencies which have a threshold of "threshold_value"

    The idea is based on values from "REDUNDANCY AND THE CHOICE OF HYDROLOGIC INDICES FOR CHARACTERIZING STREAMFLOW REGIMES
    JULIAN D. OLDEN* and N. L. POFF" (RiverResearchApp_2003.pdf, page 109). An algorithms to calculate this data is not
    given, so we developed an own.

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_value: a threshold where a baseflow event is defined with
    :type threshold_value: float
    :return: mean of deviation of average flood frequency of the best suitable section or a dict if mode was set to raw data
    :rtype: float / dict


    """
    if datetime_series is None:
        raise HydroSignaturesError("datetime_series is None. Please specify a datetime_series.")

    if data.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)
    basics.pre_process(__calcDevForFloodFrequency)
    return basics.analyze(__calcFlowLevelEventFrequency, datetime_series=datetime_series,
                          threshold_value=threshold_value,
                          flow_level_type="flood")


def getBaseflowFrequency(data, comparedata=None, mode='get_signature', datetime_series=None, threshold_value=3):
    """
    Get high and low-flow event frequencies which have a threshold of "threshold_value"

    The idea is based on values from "REDUNDANCY AND THE CHOICE OF HYDROLOGIC INDICES FOR CHARACTERIZING STREAMFLOW REGIMES
    JULIAN D. OLDEN* and N. L. POFF" (RiverResearchApp_2003.pdf, page 109). An algorithms to calculate this data is not
    given, so we developed an own.

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string
    :param datetime_series: datetime series
    :type datetime_series: pandas datetime object
    :param threshold_value: a threshold where a baseflow event is defined with
    :type threshold_value: float
    :return: mean of deviation of average flood frequency of the best suitable section or a dict if mode was set to raw data
    :rtype: float / dict


    """
    if datetime_series is None:
        raise HydroSignaturesError("datetime_series is None. Please specify a datetime_series.")

    if data.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)
    basics.pre_process(__calcDevForFloodFrequency)
    return basics.analyze(__calcFlowLevelEventFrequency, datetime_series=datetime_series,
                          threshold_value=threshold_value,
                          flow_level_type="baseflow")


def __calcDevForFloodFrequency(a, b):

    sum = 0.0
    rows_a = [row[1] for row in a.itertuples()]
    rows_b = [row[1] for row in b.itertuples()]
    for j in range(rows_a.__len__()):
        sum += _SignaturesBasicFunctionality.calcDev(rows_a[j], rows_b[j])
    return sum / b.__len__()


def __calcFlowLevelEventFrequency(data, datetime_series, threshold_value, flow_level_type):
    """
    Calc the high and low-flow event frequencies which have a threshold of "threshold_value"

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data where the flood frequency is calculated of
    :type data: list
    :param datetime_series: a pandas data object with sorted (may not be complete but sorted) dates
    :type datetime_series: pandas datetime
    :param threshold_value: a threshold where a baseflow event is defined with
    :type threshold_value: float
    :param flow_level_type: in ["flood","baseflow"]:
    :type flow_level_type: string
    :return: mean of deviation of average flood frequency of the best suitable section
    :rtype: float
    """

    if flow_level_type not in ["flood", "baseflow"]:
        raise HydroSignaturesError("flow_level_type should flood or baseflow")

    if __isSorted(datetime_series):

        count_per_section = {}

        index = 0

        s = SuitableInput(datetime_series)
        section = s.calc()

        for d in datetime_series:

            if section == "year":
                sec_key = pandas.Timestamp(datetime.datetime(year=d.to_pydatetime().year, month=1, day=1))
            elif section == "month":
                sec_key = pandas.Timestamp(datetime.datetime(year=d.to_pydatetime().year, month=d.to_pydatetime().month, day=1))
            elif section == "day":
                sec_key = d
            elif section == "hour":
                sec_key = pandas.Timestamp(datetime.datetime(year=d.to_pydatetime().year, month=d.to_pydatetime().month, day=d.to_pydatetime().day, hour=d.to_pydatetime().hour))
            else:
                raise HydroSignaturesError("Your section: " + section + " is not valid. See pydoc of this function")

            if sec_key not in count_per_section:
                count_per_section[sec_key] = 0

            if flow_level_type == "flood":
                if data[index] > threshold_value:
                    count_per_section[sec_key] += 1
            elif flow_level_type == "baseflow":
                if data[index] < threshold_value:
                    count_per_section[sec_key] += 1
            index += 1

    else:
        raise HydroSignaturesError("The time series is not sorted, so a calculation can not be performed")

    count_per_section_for_pandas = []

    for key in count_per_section:
        count_per_section_for_pandas.append({'count':count_per_section[key],'datetime':key})


    count_per_section_pandas = pandas.DataFrame(count_per_section_for_pandas)

    count_per_section_pandas = count_per_section_pandas.set_index("datetime")
    return count_per_section_pandas.sort_index()


def getLowFlowVar(data, comparedata=None, mode='get_signature', datetime_series=None):
    """

    Mean of annual minimum flow divided by the median flow (Jowett and Duncan, 1990)

    Annular Data

        .. math::

         Annualar Data= \\frac{\\sum_{i=1}^{N}(min(d_i)}{N*median(data)}

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string
    :param datetime_series: a pandas data object with sorted (may not be complete but sorted) dates
    :type datetime_series: pandas datetime object
    :return: mean of deviation of the low flow variation or a dict if mode was set to raw data
    :rtype: float / dict

    """

    if datetime_series is None:
        raise HydroSignaturesError("datetime_series is None. Please specify a datetime_series.")

    if data.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)
    return basics.analyze(__calcAnnularData, datetime_series=datetime_series,
                          what="min")




def getHighFlowVar(data, comparedata=None, mode='get_signature', datetime_series=None):
    """
    Mean of annual maximum flow divided by the median flow (Jowett and Duncan, 1990)

    Annular Data

        .. math::

         Annualar Data= \\frac{\\sum_{i=1}^{N}(max(d_i)}{N*median(data)}

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string
    :param datetime_series: a pandas data object with sorted (may not be complete but sorted) dates
    :type datetime_series: pandas datetime object
    :return: mean of deviation of the high flow variation or a dict if mode was set to raw data
    :rtype: float / dict

    """

    if datetime_series is None:
        raise HydroSignaturesError("datetime_series is None. Please specify a datetime_series.")

    if data.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)
    return basics.analyze(__calcAnnularData, datetime_series=datetime_series,
                          what="max")


def __calcAnnularData(data, datetime_series, what):
    """
    Annular Data

    :math:`Annualar Data= \\frac{\\sum_{i=1}^{N}(max(d_i)}{N}`

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)

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


def getBaseflowIndex(data, comparedata=None, mode='get_signature', datetime_series=None):
    """
    We may have to use baseflow devided with total discharge
    See https://de.wikipedia.org/wiki/Niedrigwasser and
    see also http://people.ucalgary.ca/~hayashi/kumamoto_2014/lectures/2_3_baseflow.pdf

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)
    and
    "Report No. 108, Low flow estimation in the United Kingdom, . Gustard, A. Bullock December 1992 and J. M. Dixon"
    (IH_108.pdf, page 20 ff)

    :param data: data to analyze
    :type data: list
    :param comparedata: data to analyze and compare with variable data
    :type comparedata: list
    :param mode: which mode of calculation should be used: one of get_signature, get_raw_data or calc_Dev
    :type mode: string

    :param datetime_series: a pandas data object with sorted (may not be complete but sorted) dates
    :type datetime_series: pandas datetime

    :return: deviation of base flow index
    :rtype: float
    """
    if datetime_series is None:
        raise HydroSignaturesError("datetime_series is None. Please specify a datetime_series.")

    if data.__len__() != datetime_series.__len__():
        raise HydroSignaturesError("Simulation / observation data and the datetime_series have not the same length")

    basics = _SignaturesBasicFunctionality(data, comparedata=comparedata, mode=mode)
    basics.pre_process(__calcDevForBaseflowIndex)
    return basics.analyze(__calcBaseflowIndex, datetime_series=datetime_series)


def __calcDevForBaseflowIndex(a, b):
    sum_sim = 0.0
    sum_obs = 0.0

    for y in a:
        sum_obs += a[y]
    for y in b:
        sum_sim += b[y]


    sum_obs = sum_obs / a.__len__()
    sum_sim = sum_sim / b.__len__()
    return _SignaturesBasicFunctionality.calcDev(sum_sim, sum_obs)


def __calcBaseflowIndex(data, datetime_series):
    """
    Basefow Index

    :math:`BasefowIndex = \\frac{BF}{TD}` where BF is the median of the data and TD the minimum of the data per year

    See paper "Uncertainty in hydrological signatures" by I. K. Westerberg and H. K. McMillan, Hydrol. Earth Syst. Sci.,
    19, 3951 - 3968, 2015 (hess-19-3951-2015.pdf, page 3956)
    and
    "Report No. 108, Low flow estimation in the United Kingdom, . Gustard, A. Bullock December 1992 and J. M. Dixon"
    (IH_108.pdf, page 20 ff)

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


def __help():
    print("Use .__doc__ to see description of every function")


def rel_error_list(a, b):
    """
    See See https://dsp.stackexchange.com/a/8724
    :param a:
    :param b:
    :return:
    """
    if a.__len__() == b.__len__():
        error = 0
        denominator = 0
        for i in range(a.__len__()):
            error += (a[i] - b[i]) ** 2
            denominator += a[i] * +2
        normalized = error / denominator
        return normalized
    else:
        raise Exception("Not the same length")
