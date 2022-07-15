# -*- coding: utf-8 -*-
"""
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

:paper: Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: 
SPOTting Model Parameters Using a Ready-Made Python Package, 
PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015.

Imports the different algorithms from this package.
To reduce dependencies, one may select here just the needed algorithm.
"""


from ._algorithm import _algorithm
from .abc import abc  # Artificial Bee Colony
from .dds import dds  # Dynamically Dimensioned Search algorithm
from .demcz import demcz  # Differential Evolution Markov Chain
from .dream import dream  # DiffeRential Evolution Adaptive Metropolis
from .fast import fast  # Fourier Amplitude Sensitivity Test
from .fscabc import fscabc  # Fitness Scaling Artificial Bee Colony
from .lhs import lhs  # Latin Hypercube Sampling
from .list_sampler import list_sampler  # Samples from  given spotpy database
from .mc import mc  # Monte Carlo
from .mcmc import mcmc  # Metropolis Markov Chain Monte Carlo
from .mle import mle  # Maximum Likelihood Estimation
from .nsgaii import (
    NSGAII,  # A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II
)
from .padds import padds  # Pareto Archived - Dynamicallly Dimensioned Search algorithm
from .rope import rope  # RObust Parameter Estimation
from .sa import sa  # Simulated annealing
from .sceua import sceua  # Shuffled Complex Evolution
