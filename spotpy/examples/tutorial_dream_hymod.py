# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds example code how to use the dream algorithm
'''

import numpy as np
import spotpy
#from spotpy.examples.spot_setup_hymod_exe import spot_setup
from spotpy.examples.spot_setup_hymod_python import spot_setup
import matplotlib.pyplot as plt
from  spotpy.likelihoods import gaussianLikelihoodMeasErrorOut as GausianLike
from spotpy.analyser import plot_parameter_trace
from spotpy.analyser import plot_posterior_parameter_histogram
if __name__ == "__main__":
    parallel ='seq'
    # Initialize the Hymod example (will only work on Windows systems)
    #spot_setup=spot_setup(parallel=parallel)
    spot_setup=spot_setup(GausianLike)
    
    # Create the Dream sampler of spotpy, alt_objfun is set to None to force SPOTPY
    # to jump into the def objectivefunction in the spot_setup class (default is
    # spotpy.objectivefunctions.log_p) 
    
    #Select number of maximum repetitions
    rep=5000
    
    # Select five chains and set the Gelman-Rubin convergence limit
    nChains                = 4
    convergence_limit      = 1.2
    
    # Other possible settings to modify the DREAM algorithm, for details see Vrugt (2016)
    nCr                    = 3
    eps                    = 10e-6
    runs_after_convergence = 100
    acceptance_test_option = 6
    
    sampler=spotpy.algorithms.dream(spot_setup, dbname='DREAM_hymod', dbformat='csv')
    r_hat = sampler.sample(rep, nChains, nCr, eps, convergence_limit, 
                           runs_after_convergence,acceptance_test_option)
    
    
    
    
    # Load the results gained with the dream sampler, stored in DREAM_hymod.csv
    results = spotpy.analyser.load_csv_results('DREAM_hymod')
    # Get fields with simulation data
    fields=[word for word in results.dtype.names if word.startswith('sim')]
    
    
    # Example plot to show remaining parameter uncertainty #
    fig= plt.figure(figsize=(9,6))
    ax = plt.subplot(1,1,1)
    q5,q25,q75,q95=[],[],[],[]
    for field in fields:
        q5.append(np.percentile(results[field][-100:-1],2.5))# ALl 100 runs after convergence
        q95.append(np.percentile(results[field][-100:-1],97.5))# ALl 100 runs after convergence
    ax.plot(q5,color='dimgrey',linestyle='solid')
    ax.plot(q95,color='dimgrey',linestyle='solid')
    ax.fill_between(np.arange(0,len(q5),1),list(q5),list(q95),facecolor='dimgrey',zorder=0,
                    linewidth=0,label='simulation uncertainty')  
    ax.plot(spot_setup.evaluation(),color='red', markersize=2,label='data')
    ax.set_ylim(-50,450)
    ax.set_xlim(0,729)
    ax.set_ylabel('Discharge [l s-1]')
    ax.set_xlabel('Days')
    ax.legend()
    fig.savefig('DREAM_simulation_uncertainty_Hymod.png',dpi=150)
    #########################################################
    
    
    # Example plot to show the convergence #################
    spotpy.analyser.plot_gelman_rubin(results, r_hat, fig_name='DREAM_r_hat.png')
    ########################################################
    
    
    
    
    # Example plot to show the parameter distribution ######
    parameters = spotpy.parameter.get_parameters_array(spot_setup)
    
    fig, ax = plt.subplots(nrows=5, ncols=2)
    fig.set_figheight(9)
    fig.set_figwidth(9)
    for par_id in range(len(parameters)):
        plot_parameter_trace(ax[par_id][0], results, parameters[par_id])
        plot_posterior_parameter_histogram(ax[par_id][1], results, parameters[par_id])
    
    ax[-1][0].set_xlabel('Iterations')
    ax[-1][1].set_xlabel('Parameter range')
    
    plt.show()
    fig.tight_layout()
    fig.savefig('DREAM_parameter_uncertainty_Hymod.png',dpi=300)
    #######################################################