# -*- coding: utf-8 -*-
'''
Copyright (c) 2017 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Benjamin Manns

This code shows you, how to use the hydroligcal signatures. They can also be implemented in the def objective function.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from spotpy.examples.spot_setup_hymod import spot_setup
import spotpy.signatures as sig

from pprint import pprint
print("INFO: For this example you need the folder >hymod< in the examples folder")

spot_setup = spot_setup()
parameterset = spot_setup.parameters()['random']
simulation = spot_setup.simulation(parameterset)
observation = spot_setup.evaluation()


# Beispiele zum einfachen Copy & Paste

print(sig.getMeanFlow(simulation, observation,mode="get_signature"))
print(sig.getMeanFlow(simulation, observation,mode="get_raw_data"))
print(sig.getMeanFlow(simulation, observation,mode="calc_Dev"))

print(sig.getMedianFlow(simulation, observation,mode="get_signature"))
print(sig.getMedianFlow(simulation, observation,mode="get_raw_data"))
print(sig.getMedianFlow(simulation, observation,mode="calc_Dev"))

print(sig.getSkewness(simulation, observation,mode="get_signature"))
print(sig.getSkewness(simulation, observation,mode="get_raw_data"))
print(sig.getSkewness(simulation, observation,mode="calc_Dev"))

print(sig.getCoeffVariation(simulation, observation,mode="get_signature"))
print(sig.getCoeffVariation(simulation, observation,mode="get_raw_data"))
print(sig.getCoeffVariation(simulation, observation,mode="calc_Dev"))

print(sig.getQ001(simulation, observation,mode="get_signature"))
print(sig.getQ001(simulation, observation,mode="get_raw_data"))
print(sig.getQ001(simulation, observation,mode="calc_Dev"))

print(sig.getQ01(simulation, observation,mode="get_signature"))
print(sig.getQ01(simulation, observation,mode="get_raw_data"))
print(sig.getQ01(simulation, observation,mode="calc_Dev"))

print(sig.getQ1(simulation, observation,mode="get_signature"))
print(sig.getQ1(simulation, observation,mode="get_raw_data"))
print(sig.getQ1(simulation, observation,mode="calc_Dev"))

print(sig.getQ5(simulation, observation,mode="get_signature"))
print(sig.getQ5(simulation, observation,mode="get_raw_data"))
print(sig.getQ5(simulation, observation,mode="calc_Dev"))

print(sig.getQ10(simulation, observation,mode="get_signature"))
print(sig.getQ10(simulation, observation,mode="get_raw_data"))
print(sig.getQ10(simulation, observation,mode="calc_Dev"))

print(sig.getQ20(simulation, observation,mode="get_signature"))
print(sig.getQ20(simulation, observation,mode="get_raw_data"))
print(sig.getQ20(simulation, observation,mode="calc_Dev"))

print(sig.getQ85(simulation, observation,mode="get_signature"))
print(sig.getQ85(simulation, observation,mode="get_raw_data"))
print(sig.getQ85(simulation, observation,mode="calc_Dev"))

print(sig.getQ95(simulation, observation,mode="get_signature"))
print(sig.getQ95(simulation, observation,mode="get_raw_data"))
print(sig.getQ95(simulation, observation,mode="calc_Dev"))

print(sig.getQ99(simulation, observation,mode="get_signature"))
print(sig.getQ99(simulation, observation,mode="get_raw_data"))
print(sig.getQ99(simulation, observation,mode="calc_Dev"))

print(sig.getSlopeFDC(simulation, observation,mode="get_signature"))
print(sig.getSlopeFDC(simulation, observation,mode="get_raw_data"))
print(sig.getSlopeFDC(simulation, observation,mode="calc_Dev"))

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    timespanlen = simulation.__len__()
    ddd = pd.date_range("2015-01-01 11:00", freq="5min",periods=timespanlen)
    dd_daily = pd.date_range("2015-05-01", periods=timespanlen)

    print(sig.getAverageFloodOverflowPerSection(simulation, observation,mode="get_signature", datetime_series=dd_daily,threshold_value=1))
    print(sig.getAverageFloodOverflowPerSection(simulation, observation,mode="get_raw_data", datetime_series=dd_daily,threshold_value=1))
    print(sig.getAverageFloodOverflowPerSection(simulation, observation,mode="calc_Dev", datetime_series=dd_daily,threshold_value=1))

    print(sig.getAverageFloodFrequencyPerSection(simulation, observation,datetime_series=dd_daily,threshold_value=1,mode="get_signature"))
    print(sig.getAverageFloodFrequencyPerSection(simulation, observation,datetime_series=dd_daily,threshold_value=1,mode="get_raw_data"))

    # If you want to plot the raw data, simple do:
    vals = sig.getAverageFloodFrequencyPerSection(simulation, observation,datetime_series=dd_daily,threshold_value=1,mode="get_raw_data")
    vals.plot()
    plt.show()

    print(sig.getAverageFloodFrequencyPerSection(simulation, observation,datetime_series=dd_daily,threshold_value=1,mode="calc_Dev"))

    print(sig.getAverageFloodDuration(simulation, observation,datetime_series=dd_daily,threshold_value=3,mode="get_signature"))
    print(sig.getAverageFloodDuration(simulation, observation,datetime_series=dd_daily,threshold_value=3,mode="get_raw_data"))
    print(sig.getAverageFloodDuration(simulation, observation,datetime_series=dd_daily,threshold_value=3,mode="calc_Dev"))

    print(sig.getAverageBaseflowUnderflowPerSection(simulation, observation,datetime_series=dd_daily,threshold_value=4,mode="get_signature"))
    print(sig.getAverageBaseflowUnderflowPerSection(simulation, observation,datetime_series=dd_daily,threshold_value=4,mode="get_raw_data"))
    print(sig.getAverageBaseflowUnderflowPerSection(simulation, observation,datetime_series=dd_daily,threshold_value=4,mode="calc_Dev"))

    print(sig.getAverageBaseflowFrequencyPerSection(simulation, observation,datetime_series=dd_daily,threshold_value=3,mode="get_signature"))
    print(sig.getAverageBaseflowFrequencyPerSection(simulation, observation,datetime_series=ddd,threshold_value=5,mode="get_raw_data"))
    print(sig.getAverageBaseflowFrequencyPerSection(simulation, observation,datetime_series=dd_daily,threshold_value=3,mode="calc_Dev"))

    print(sig.getAverageBaseflowDuration(simulation, observation, datetime_series=dd_daily, threshold_value=3,mode="get_signature"))
    print(sig.getAverageBaseflowDuration(simulation, observation, datetime_series=ddd, threshold_value=5,mode="get_raw_data"))
    print(sig.getAverageBaseflowDuration(simulation, observation, datetime_series=dd_daily, threshold_value=3,mode="calc_Dev"))



    print(sig.getFloodFrequency(simulation, observation,datetime_series=pd.date_range("2015-05-01", periods=timespanlen),threshold_value=3,mode="get_signature"))
    print(sig.getFloodFrequency(simulation, observation,datetime_series=pd.date_range("2015-05-01", periods=timespanlen),threshold_value=3,mode="get_raw_data"))
    print(sig.getFloodFrequency(simulation, observation,datetime_series=pd.date_range("2015-05-01", periods=timespanlen),threshold_value=3,mode="calc_Dev"))

    print(sig.getBaseflowFrequency(simulation, observation,datetime_series=pd.date_range("2015-05-01", periods=timespanlen),threshold_value=3,mode="get_signature"))
    print(sig.getBaseflowFrequency(simulation, observation,datetime_series=pd.date_range("2015-05-01", freq="5min", periods=timespanlen),threshold_value=3,mode="get_raw_data"))
    print(sig.getBaseflowFrequency(simulation, observation,datetime_series=pd.date_range("2015-05-01", periods=timespanlen),threshold_value=3,mode="calc_Dev"))

    print(sig.getLowFlowVar(simulation, observation, datetime_series=pd.date_range("2015-05-01", periods=timespanlen),mode="get_signature"))
    print(sig.getLowFlowVar(simulation, observation, datetime_series=pd.date_range("2015-05-01", periods=timespanlen),mode="get_raw_data"))
    print(sig.getLowFlowVar(simulation, observation, datetime_series=pd.date_range("2015-05-01", periods=timespanlen),mode="calc_Dev"))

    print(sig.getHighFlowVar(simulation, observation, datetime_series=pd.date_range("2015-05-01", periods=timespanlen),mode="get_signature"))
    print(sig.getHighFlowVar(simulation, observation, datetime_series=pd.date_range("2015-05-01", periods=timespanlen),mode="get_raw_data"))
    print(sig.getHighFlowVar(simulation, observation, datetime_series=pd.date_range("2015-05-01", periods=timespanlen),mode="calc_Dev"))

    print(sig.getBaseflowIndex(simulation, observation, datetime_series=pd.date_range("2015-05-01", periods=timespanlen),mode="get_signature"))
    print(sig.getBaseflowIndex(simulation, observation, datetime_series=pd.date_range("2015-05-01", periods=timespanlen),mode="get_raw_data"))
    print(sig.getBaseflowIndex(simulation, observation, datetime_series=pd.date_range("2015-05-01", periods=timespanlen),mode="calc_Dev"))

except ImportError:
    print('Please install Pandas to use these signature functions')



