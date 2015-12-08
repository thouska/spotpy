# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This package enables the comprehensive use of different Bayesian and Heuristic calibration 
techniques in one Framework. It comes along with an algorithms folder for the 
sampling and an analyser class for the plotting of results by the sampling.

:dependencies: - Numpy >1.8 (http://www.numpy.org/) 
               - Pandas >0.13 (http://pandas.pydata.org/)
               - Scipy >0.14 (http://www.scipy.org/) 
               - Matplotlib >1.4 (http://matplotlib.org/) 
               - CMF (optional) (http://fb09-pasig.umwelt.uni-giessen.de:8000/)
               - mpi4py (optional) (http://mpi4py.scipy.org/)
:help: For specific questions, try to use the socumentation website at:
http://fb09-pasig.umwelt.uni-giessen.de/spotpy/

For general things about parameter optimization techniques have a look at:
https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/
'''

import database
import algorithms
#import analyser #Acitivate if you want to have the analyser imported by default (not recommended for High Performance Computing Clusters, because of dependencies on Matplotlib)
import objectivefunctions
import parameter
__version__ = '1.1.4'
