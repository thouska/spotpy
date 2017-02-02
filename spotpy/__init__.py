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

               :help: For specific questions, try to use the socumentation website at:
http://fb09-pasig.umwelt.uni-giessen.de/spotpy/

For general things about parameter optimization techniques have a look at:
https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/

Pleas cite our paper, if you are using SPOTPY.
'''

import database
import algorithms
import analyser #Acitivate if you want to have the analyser imported by default (not recommended for High Performance Computing Clusters, because of dependencies on Matplotlib)
import objectivefunctions
import parameter
import examples
__version__ = '1.2.32'
