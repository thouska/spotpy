# -*- coding: utf-8 -*-
"""
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Anna Herzog, Tobias Houska

This class holds example code how to use the eFAST algorithm, which was adapted from the R fast package by Reusser et al. 2011
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
    rep = 100

    # Start a sensitivity analysis
    sampler = spotpy.algorithms.efast(
        spot_setup, dbname="eFAST_hymod", dbformat="csv", db_precision=np.float32
    )
    sampler.sample(rep)

    # Load the results gained with the fast sampler, stored in FAST_hymod.csv
    results = spotpy.analyser.load_csv_results("eFAST_hymod")

    # calculate the sensitivities
    sampler.calc_sensitivity(results, dbname = "eFAST_sens_hymod")

    # plot the temporal parameter sensitivities
    spotpy.analyser.plot_efast(dbname="eFast_sens_hymod")
