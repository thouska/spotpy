# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

Imports the different algorithms from this package.
To reduce dependencies, one may select here just the needed algorithm.
'''

from _algorithm import _algorithm
from demcz import demcz  # Differential Evolution Markov Chain Monte Carlo
from lhs import lhs      # Latin Hypercube Sampling
from mcmc import mcmc    # Metropolis Markov Chain Monte Carlo
from mle import mle      # Maximum Likelihood Estimation
from mc import mc        # Monte Carlo
from sceua import sceua  # Shuffled Complex Evolution
from sa import sa        # Simulated annealing
from rope import rope    # RObust Parameter Estimation
from fast import fast    # Fourier Amplitude Sensitivity Test

