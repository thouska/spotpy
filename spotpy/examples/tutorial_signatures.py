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
import pandas as pd
import spotpy.signatures as sig

print("INFO: For this example you need the folder >hymod< in the examples folder")

spot_setup = spot_setup()
parameterset = spot_setup.parameters()['random']
simulation = spot_setup.simulation(parameterset)
observation = spot_setup.evaluation()

timespanlen = simulation.__len__()
ddd = pd.date_range("2015-01-01 11:00", freq="5min",periods=timespanlen)
dd_daily = pd.date_range("2015-05-01", periods=timespanlen)


print(sig.getMeanFlow(simulation, observation))
print(sig.getMedianFlow(simulation, observation))
print(sig.getSkewness(simulation, observation))
print(sig.getCoeffVariation(simulation, observation))
print(sig.getQ001(simulation, observation))
print(sig.getQ01(simulation, observation))
print(sig.getQ1(simulation, observation))
print(sig.getQ5(simulation, observation))
print(sig.getQ10(simulation, observation))
print(sig.getQ20(simulation, observation))
print(sig.getQ85(simulation, observation))
print(sig.getQ95(simulation, observation))
print(sig.getQ99(simulation, observation))
print(sig.getAverageFloodOverflowPerSection(simulation, observation,dd_daily,threshold_factor=1, section="day"))
print(sig.getAverageFloodFrequencyPerSection(simulation, observation,dd_daily,threshold_factor=1, section="day"))
print(sig.getAverageFloodDuration(simulation, observation,dd_daily,threshold_factor=3, section="day"))
print(sig.getAverageBaseflowUnderflowPerSection(simulation, observation,dd_daily,threshold_factor=4, section="day"))
print(sig.getAverageBaseflowFrequencyPerSection(simulation, observation,dd_daily,3, "day"))
print(sig.getAverageBaseflowDuration(simulation, observation,dd_daily,threshold_factor=0.2, section="day"))
print(sig.getFloodFrequency(simulation, observation,pd.date_range("2015-05-01", periods=timespanlen),3, "day"))
print(sig.getBaseflowFrequency(simulation, observation,pd.date_range("2015-05-01", periods=timespanlen),3, "day"))
print(sig.getLowFlowVar(simulation, observation, pd.date_range("2015-05-01", periods=timespanlen)))
print(sig.getHighFlowVar(simulation, observation, pd.date_range("2015-05-01", periods=timespanlen)))
print(sig.getBaseflowIndex(simulation, observation, pd.date_range("2015-05-01", periods=timespanlen)))
print(sig.getSlopeFDC(simulation, observation))


