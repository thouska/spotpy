# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds example code how to use the dream algorithm
'''

import spotpy
from spotpy.examples.spot_setup_hymod_exe import spot_setup
#from spotpy.examples.spot_setup_hymod_python import spot_setup


if __name__ == "__main__":
    parallel ='seq'
    # Initialize the Hymod example
    spot_setup=spot_setup()
    
    #Select number of maximum repetitions
    rep=1000
    
    #Start a sensitivity analysis
    sampler=spotpy.algorithms.fast(spot_setup, dbname='FAST_hymod', dbformat='csv')
    #sampler.sample(rep)
    
    # Load the results gained with the dream sampler, stored in DREAM_hymod.csv
    results = spotpy.analyser.load_csv_results('FAST_hymod')
    
    # Example plot to show the sensitivity index of each parameter
    spotpy.analyser.plot_fast_sensitivity(results, number_of_sensitiv_pars=3)

    
