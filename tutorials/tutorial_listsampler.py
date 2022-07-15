"""
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This example implements the Rosenbrock function into SPOT.  
"""

import spotpy
from spotpy.examples.spot_setup_hymod_python import spot_setup

if __name__ == "__main__":
    # 1 We start classical by perfroming a sensitivity analysis
    parallel = "seq"
    # 1.1  Initialize the Hymod example
    spot_setup = spot_setup()

    # S 1.2 elect number of maximum repetitions
    rep = 1345

    # 1.3 Start a sensitivity analysis
    sampler = spotpy.algorithms.fast(spot_setup, dbname="FAST_hymod", dbformat="csv")
    sampler.sample(rep)
    ###########################################################

    # 2 Special part begins

    # 2.1 Lets assume something went wrong with the model or you want to use
    # the same parameter set again. Then the list_sampler is what you need.
    # It is also thinkable that you want to change something in this database
    # and restart.
    # The list_sampler takes a spotpy database as input, reads the paramater
    # in it and starts the model again:

    # 2.2 Use the generated database to reuse the parameters:
    sampler_new = spotpy.algorithms.list_sampler(
        spot_setup, dbname="FAST_hymod", dbformat="csv"
    )

    # 2.3 Start the sampler with the according number of repetitions
    sampler_new.sample(rep)

    # 2.4 Load the result, the new database will have added the word "list"
    results = spotpy.analyser.load_csv_results("FAST_hymodlist")

    # 2.5 Please mind that these results contains the same data as we have used
    # the same spot_setup. However we could have also used another spot_setup,
    # which takes the same parameters.
    #############################################################

    # 3.0 Classical part starts again

    # 3.1 Example plot to show the sensitivity index of each parameter from t
    spotpy.analyser.plot_fast_sensitivity(results, number_of_sensitiv_pars=3)

    # 3.2 Example to get the sensitivity index of each parameter
    SI = spotpy.analyser.get_sensitivity_of_fast(results)
    #############################################################
