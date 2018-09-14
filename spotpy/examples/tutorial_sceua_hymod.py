# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds example code how to use the dream algorithm
'''

import numpy as np
import spotpy
from spotpy.examples.spot_setup_hymod_exe import spot_setup
#from spotpy.examples.spot_setup_hymod_python import spot_setup
import pylab as plt


if __name__ == "__main__":
    parallel ='seq'
    # Initialize the Hymod example (will only work on Windows systems)
    #spot_setup=spot_setup(parallel=parallel)
    spot_setup=spot_setup()
    
    # Create the Dream sampler of spotpy, al_objfun is set to None to force SPOTPY
    # to jump into the def objectivefunction in the spot_setup class (default is
    # spotpy.objectivefunctions.log_p) 
    
    #Select number of maximum repetitions
    rep=20
    
    # Select five chains and set the Gelman-Rubin convergence limit
    nChains                = 4
    convergence_limit      = 1.2
    runs_after_convergence = 100
    np.random.seed(42)
    sampler=spotpy.algorithms.sceua(spot_setup, dbname='SCEUA_hymod', dbformat='csv', alt_objfun=None)
    sampler.sample(rep)
    
    
    
    
    # Load the results gained with the dream sampler, stored in DREAM_hymod.csv
    results = spotpy.analyser.load_csv_results('SCEUA_hymod')
    print(results['parcmax'][0:10])
    # Get fields with simulation data
    fields=[word for word in results.dtype.names if word.startswith('sim')]
    
    
    # Example plot to show remaining parameter uncertainty #
    fig= plt.figure(figsize=(16,9))
    ax = plt.subplot(1,1,1)
    q5,q25,q75,q95=[],[],[],[]
    for field in fields:
        q5.append(np.percentile(results[field][-100:-1],2.5))
        q95.append(np.percentile(results[field][-100:-1],97.5))
    #ax.plot(q5,color='dimgrey',linestyle='solid')
    #ax.plot(q95,color='dimgrey',linestyle='solid')
    #ax.fill_between(np.arange(0,len(q5),1),list(q5),list(q95),facecolor='dimgrey',zorder=0,
    #                linewidth=0,label='parameter uncertainty')  
    ax.plot(spot_setup.evaluation(),'r.',label='data')
    ax.set_ylim(-50,450)
    ax.set_xlim(0,729)
    ax.legend()
    fig.savefig('python_hymod.png',dpi=300)
    #########################################################
    
    

    
