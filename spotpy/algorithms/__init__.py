# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

:paper: Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: 
SPOTting Model Parameters Using a Ready-Made Python Package, 
PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015.

Imports the different algorithms from this package.
To reduce dependencies, one may select here just the needed algorithm.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
from . import demcz     # Differential Evolution Markov Chain Monte Carlo
from . import lhs         # Latin Hypercube Sampling
from . import mcmc       # Metropolis Markov Chain Monte Carlo
from . import mle         # Maximum Likelihood Estimation
from . import mc           # Monte Carlo
from . import sceua     # Shuffled Complex Evolution
from . import sa           # Simulated annealing
from . import rope       # RObust Parameter Estimation
from . import fast       # Fourier Amplitude Sensitivity Test
from . import abc         # Artificial Bee Colony
from . import fscabc   # Fitness Scaling Artificial Bee Colony
