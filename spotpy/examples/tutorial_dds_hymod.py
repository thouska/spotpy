# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds example code how to use the dream algorithm
'''

import numpy as np
try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy

from spotpy.examples.spot_setup_hymod_python import spot_setup
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parallel ='seq' # Runs everthing in sequential mode
    np.random.seed(2000) # Makes the results reproduceable
    
    # Initialize the Hymod example
    # In this case, we tell the setup which algorithm we want to use, so
    # we can use this exmaple for different algorithms
    spot_setup=spot_setup(users_objective_function=spotpy.objectivefunctions.nashsutcliffe)
    
    #Select number of maximum allowed repetitions
    rep=1000
        
    # Create the SCE-UA sampler of spotpy, alt_objfun is set to None to force SPOTPY
    # to jump into the def objectivefunction in the spot_setup class (default is
    # spotpy.objectivefunctions.rmse) 
    sampler=spotpy.algorithms.dds(spot_setup, dbname='DDS_hymod', dbformat='csv')
    
    #Start the sampler, one can specify ngs, kstop, peps and pcento id desired
    sampler.sample(rep)
    results = sampler.getdata()
    
    fig= plt.figure(1,figsize=(9,5))
    plt.plot(results['like1'])
    plt.show()
    plt.ylabel('Objective function')
    plt.xlabel('Iteration')
    fig.savefig('hymod_objectivefunction.png',dpi=300)
    
    # Example plot to show the parameter distribution ###### 
    def plot_parameter_trace(ax, results, parameter):
        ax.plot(results['par'+parameter.name],'.')
        ax.set_ylabel(parameter.name)
        ax.set_ylim(parameter.minbound, parameter.maxbound)
        
    def plot_parameter_histogram(ax, results, parameter):
        #chooses the last 20% of the sample
        ax.hist(results['par'+parameter.name][int(len(results)*0.9):], 
                 bins =np.linspace(parameter.minbound,parameter.maxbound,20))
        ax.set_ylabel('Density')
        ax.set_xlim(parameter.minbound, parameter.maxbound)
        
    fig= plt.figure(2,figsize=(9,9))
    
    ax1 = plt.subplot(5,2,1)
    plot_parameter_trace(ax1, results, spot_setup.cmax)
    
    ax2 = plt.subplot(5,2,2)
    plot_parameter_histogram(ax2,results, spot_setup.cmax)
    
    ax3 = plt.subplot(5,2,3)
    plot_parameter_trace(ax3, results, spot_setup.bexp)
    
    ax4 = plt.subplot(5,2,4)
    plot_parameter_histogram(ax4, results, spot_setup.bexp)
        
    ax5 = plt.subplot(5,2,5)
    plot_parameter_trace(ax5, results, spot_setup.alpha)
     
    ax6 = plt.subplot(5,2,6)
    plot_parameter_histogram(ax6, results, spot_setup.alpha)

    ax7 = plt.subplot(5,2,7)
    plot_parameter_trace(ax7, results, spot_setup.Ks)
    
    ax8 = plt.subplot(5,2,8)
    plot_parameter_histogram(ax8, results, spot_setup.Ks)

    ax9 = plt.subplot(5,2,9)
    plot_parameter_trace(ax9, results, spot_setup.Kq)
    ax9.set_xlabel('Iterations')
    
    ax10 = plt.subplot(5,2,10)
    plot_parameter_histogram(ax10, results, spot_setup.Kq)
    ax10.set_xlabel('Parameter range')
    
    plt.show()
    fig.savefig('hymod_parameters.png',dpi=300)