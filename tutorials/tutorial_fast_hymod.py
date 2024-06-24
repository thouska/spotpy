# -*- coding: utf-8 -*-
"""
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds example code how to use the FAST algorithm
"""

import numpy as np

import spotpy
from spotpy.examples.spot_setup_hymod_python import spot_setup

if __name__ == "__main__":
    parallel = "seq"
    # Initialize the Hymod example
    spot_setup = spot_setup()

    # Select number of maximum repetitions
    # CHeck out https://spotpy.readthedocs.io/en/latest/Sensitivity_analysis_with_FAST/
    # How to determine an appropriate number of repetitions
    rep = 2245

    # Start a sensitivity analysis
    sampler = spotpy.algorithms.fast(
        spot_setup, dbname="FAST_hymod", dbformat="csv", db_precision=np.float32
    )
    sampler.sample(rep)

    # Load the results gained with the fast sampler, stored in FAST_hymod.csv
    results = spotpy.analyser.load_csv_results("FAST_hymod")

    # Example plot to show the sensitivity index of each parameter
    spotpy.analyser.plot_fast_sensitivity(results, number_of_sensitiv_pars=3)

    # Example to get the sensitivity index of each parameter
    SI = spotpy.analyser.get_sensitivity_of_fast(results)
