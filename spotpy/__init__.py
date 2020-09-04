# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

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
               - sqlite3 (optional) (https://pypi.python.org/pypi/sqlite3/)
               - numba (optional) (https://pypi.python.org/pypi/numba/)

               :help: For specific questions, try to use the documentation website at:
                https://spotpy.readthedocs.io/en/latest/

For general things about parameter optimization techniques have a look at:
https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/

Please cite our paper, if you are using SPOTPY.
'''
from . import database            # Writes the results of the sampler in a user defined output file
from . import algorithms          # Contains all the different algorithms implemented in SPOTPY 
from . import parameter           # Contains different distributions to describe the prior information for every model parameter
from . import analyser            # Contains some examples to analyse the results of the different algorithms
from . import objectivefunctions  # Quantifies goodness of fit between simulation and evaluation data with objective functions
from . import likelihoods         # Quantifies goodness of fit between simulation and evaluation data with likelihood functions
from . import examples            # Contains tutorials how to use SPOTPY
from . import describe            # Contains some helper functions to describe samplers and set-ups
from .hydrology import signatures # Quantifies goodness of fit between simulation and evaluation data with hydrological signatures

__version__ = '1.5.11'