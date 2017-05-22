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
import pandas as pd
import spotpy.signatures as sig

print("INFO: For this example you need the folder >hymod< in the examples folder")

spot_setup = spot_setup()
parameterset = spot_setup.parameters()['random']
simulation = spot_setup.simulation(parameterset)
observation = spot_setup.evaluation()

timespanlen = simulation.__len__()

print(sig.getMedianFlow(simulation, observation))
print(sig.getDuration(simulation, observation,pd.date_range("2015-05-01", periods=timespanlen),0.2))
print(sig.getBaseflowIndex(simulation, observation, pd.date_range("2015-05-01", periods=timespanlen)))
print(sig.getSlopeFDC(simulation, observation))
print(sig.getLowFlowVar(simulation, observation, pd.date_range("2015-05-01", periods=timespanlen)))
print(sig.getHighFlowVar(simulation, observation, pd.date_range("2015-05-01", periods=timespanlen)))
