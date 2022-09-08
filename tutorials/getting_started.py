# -*- coding: utf-8 -*-
"""
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds the example code from the getting_started web-documention.
"""
# Getting started

# To start your experience with SPOTPY you need to have SPOTPY installed.
# Please see the [Installation chapter](index.md) for further details.
# To use SPOTPY we have to import it and use one of the pre-build examples:
import spotpy  # Load the SPOTPY package into your working storage
from spotpy.examples.spot_setup_rosenbrock import spot_setup

# The example comes along with parameter boundaries, the Rosenbrock function,
# the optimal value of the function and RMSE as a likelihood.
# So we can directly start to analyse the Rosenbrock function with one of the
# algorithms. We start with a simple Monte Carlo sampling

if __name__ == "__main__":
    # Give Monte Carlo algorithm the example setup and saves results in a
    # RosenMC.csv file
    sampler = spotpy.algorithms.mc(spot_setup(), dbname="RosenMC", dbformat="ram")

    # Now we can sample with the implemented Monte Carlo algortihm:
    sampler.sample(50000)  # Sample 50.000 parameter combinations
    results = sampler.getdata()
    # Now we want to have a look at the results. First we want to know,
    # what the algorithm has done during the 50.000 iterations:
    spotpy.analyser.plot_parameterInteraction(results)
    # Now we collect the 10% runs with the lowest objective function
    posterior = spotpy.analyser.get_posterior(results, maximize=False)
    spotpy.analyser.plot_parameterInteraction(posterior)
    # Print the run with the lowest objective function
    print(spotpy.analyser.get_best_parameterset(results, maximize=False))
