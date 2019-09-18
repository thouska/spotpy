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




from ._algorithm import _algorithm
from .demcz import demcz     # Differential Evolution Markov Chain
from .lhs import lhs         # Latin Hypercube Sampling
from .mcmc import mcmc       # Metropolis Markov Chain Monte Carlo
from .mle import mle         # Maximum Likelihood Estimation
from .mc import mc           # Monte Carlo
from .sceua import sceua     # Shuffled Complex Evolution
from .sa import sa           # Simulated annealing
from .rope import rope       # RObust Parameter Estimation
from .fast import fast       # Fourier Amplitude Sensitivity Test
from .abc import abc         # Artificial Bee Colony
from .fscabc import fscabc   # Fitness Scaling Artificial Bee Colony
from .dream import dream     # DiffeRential Evolution Adaptive Metropolis
from .list_sampler import list_sampler  # Samples from  given spotpy database
from .dds import dds         # Dynamically Dimensioned Search algorithm
from .padds import padds     # Pareto Archived - Dynamicallly Dimensioned Search algorithm
