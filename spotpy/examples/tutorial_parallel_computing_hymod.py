# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds example code how to use the dream algorithm
'''

try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy
#from spotpy.examples.spot_setup_hymod_python import spot_setup
import sys

# When you are using parallel = 'mpc' you need to have 
#the if __name__ == "__main__": line in your script
if __name__ == "__main__":
    rep = 200
    #If you are working on a windows computer, this will be True
    if 'win' in sys.platform:
        parallel ='mpc'
        from spotpy.examples.spot_setup_hymod_exe import spot_setup

   
   # If not you probably have a Mac or Unix system. Then save this file and start it
   # from your terminal with "mpirun -c 20 python your_script.py"
    else:
        parallel = 'mpi'
        from spotpy.examples.spot_setup_hymod_unix import spot_setup

    # Initialize the Hymod example (this runs on Windows systems only)
    # Checkout the spot_setup_hymod_exe.py to see how you need to adopt
    # your spot_setup class to run in parallel
    # If your model in def simulation reads any files, make sure, they are
    # unique for each cpu. Otherwise things get messed up...
    spot_setup=spot_setup(parallel=parallel)
    
    # Initialize a sampler that is suited for parallel computing (all except MLE, MCMC and SA)
    sampler=spotpy.algorithms.mc(spot_setup, dbname='Parallel_hymod', dbformat='csv',
                                 parallel=parallel) 
    # Sample in parlallel
    sampler.sample(rep)
    
    # Load results from file
    #results = spotpy.analyser.load_csv_results('Parallel_hymod')
    
    # Plot best model run
    #spotpy.analyser.plot_bestmodelrun(results,spot_setup.evaluation())