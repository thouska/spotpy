# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds example code how to use the dream algorithm
'''

import numpy as np
import spotpy
from spotpy.examples.spot_setup_hymod_python import spot_setup
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parallel ='seq' # Runs everthing in sequential mode
    np.random.seed(2000) # Makes the results reproduceable
    
    # Initialize the Hymod example
    # In this case, we tell the setup which algorithm we want to use, so
    # we can use this exmaple for different algorithms
    spot_setup=spot_setup(spotpy.objectivefunctions.rmse)
    
    #Select number of maximum allowed repetitions
    rep=5000
    sampler=spotpy.algorithms.sceua(spot_setup, dbname='SCEUA_hymod', dbformat='csv')
    
    #Start the sampler, one can specify ngs, kstop, peps and pcento id desired
    sampler.sample(rep, ngs=7, kstop=3, peps=0.1, pcento=0.1) 
    
    # Load the results gained with the sceua sampler, stored in SCEUA_hymod.csv
    results = spotpy.analyser.load_csv_results('SCEUA_hymod')
    
    
    #Plot how the objective function was minimized during sampling
    fig= plt.figure(1,figsize=(9,6))
    plt.plot(results['like1'])
    plt.show()
    plt.ylabel('RMSE')
    plt.xlabel('Iteration')
    fig.savefig('SCEUA_objectivefunctiontrace.png',dpi=150)
    
    # Plot the best model run
    #Find the run_id with the minimal objective function value
    bestindex,bestobjf = spotpy.analyser.get_minlikeindex(results)

    # Select best model run
    best_model_run = results[bestindex]
    
    #Filter results for simulation results
    fields=[word for word in best_model_run.dtype.names if word.startswith('sim')]
    best_simulation = list(best_model_run[fields])

    fig= plt.figure(figsize=(9,6))
    ax = plt.subplot(1,1,1)
    ax.plot(best_simulation,color='black',linestyle='solid', label='Best objf.='+str(bestobjf))
    ax.plot(spot_setup.evaluation(),'r.',markersize=3, label='Observation data')
    plt.xlabel('Number of Observation Points')
    plt.ylabel ('Discharge [l s-1]')
    plt.legend(loc='upper right')
    fig.savefig('SCEUA_best_modelrun.png',dpi=150)

