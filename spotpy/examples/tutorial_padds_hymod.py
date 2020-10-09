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


def multi_obj_func(evaluation, simulation):
    #used to overwrite objective function in hymod example
    like1 = abs(spotpy.objectivefunctions.pbias(evaluation, simulation))
    like2 = spotpy.objectivefunctions.rmse(evaluation, simulation)
    like3 = spotpy.objectivefunctions.rsquared(evaluation, simulation)*-1
    return np.array([like1, like2, like3])

if __name__ == "__main__":
    parallel ='seq' # Runs everthing in sequential mode
    np.random.seed(2000) # Makes the results reproduceable
    
    # Initialize the Hymod example
    # In this case, we tell the setup which algorithm we want to use, so
    # we can use this exmaple for different algorithms
    spot_setup=spot_setup(multi_obj_func)

    #Select number of maximum allowed repetitions
    rep=2000
    
        
    # Create the PADDS sampler of spotpy, alt_objfun is set to None to force SPOTPY
    # to jump into the def objectivefunction in the spot_setup class (default is
    # spotpy.objectivefunctions.rmse) 
    sampler=spotpy.algorithms.padds(sp_setup, dbname='padds_hymod', dbformat='csv')
    
    #Start the sampler, one can specify metric if desired
    sampler.sample(rep,metric='ones')
    
    # Load the results gained with the sceua sampler, stored in padds_hymod.csv
    #results = spotpy.analyser.load_csv_results('padds_hymod')
    results = sampler.getdata()

    # from pprint import pprint
    # #pprint(results)
    # pprint(results['chain'])

    for likno in range(1,4):
        fig_like1 = plt.figure(1,figsize=(9,5))
        plt.plot(results['like'+str(likno)])
        plt.show()
        fig_like1.savefig('hymod_padds_objectivefunction_' + str(likno) + '.png', dpi=300)


    fig, ax=plt.subplots(3)
    for likenr in range(3):
        ax[likenr].plot(results['like'+str(likenr+1)])
        ax[likenr].set_ylabel('like'+str(likenr+1))
    fig.savefig('hymod_padds_objectivefunction.png', dpi=300)




    
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
    
    
    # Example plot to show remaining parameter uncertainty #
    fields=[word for word in results.dtype.names if word.startswith('sim')]
    fig= plt.figure(3, figsize=(16,9))
    ax11 = plt.subplot(1,1,1)
    q5,q25,q75,q95=[],[],[],[]
    for field in fields:
        q5.append(np.percentile(results[field][-100:-1],2.5))
        q95.append(np.percentile(results[field][-100:-1],97.5))
    ax11.plot(q5,color='dimgrey',linestyle='solid')
    ax11.plot(q95,color='dimgrey',linestyle='solid')
    ax11.fill_between(np.arange(0,len(q5),1),list(q5),list(q95),facecolor='dimgrey',zorder=0,
                    linewidth=0,label='parameter uncertainty')
    ax11.plot(spot_setup.evaluation(),'r.',label='data')
    ax11.set_ylim(-50,450)
    ax11.set_xlim(0,729)
    ax11.legend()
    fig.savefig('python_hymod.png',dpi=300)
    #########################################################