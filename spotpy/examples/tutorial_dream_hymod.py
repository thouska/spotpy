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

if __name__ == "__main__":
    parallel ='seq'
    # Initialize the Hymod example (will only work on Windows systems)
    #spot_setup=spot_setup(parallel=parallel)
    spot_setup=spot_setup(GausianLike)
    
    # Create the Dream sampler of spotpy, al_objfun is set to None to force SPOTPY
    # to jump into the def objectivefunction in the spot_setup class (default is
    # spotpy.objectivefunctions.log_p) 
    
    #Select number of maximum repetitions
    rep=5000
    
    # Select five chains and set the Gelman-Rubin convergence limit
    nChains                = 4
    convergence_limit      = 1.2
    runs_after_convergence = 100
    
    sampler=spotpy.algorithms.dream(spot_setup, dbname='DREAM_hymod', dbformat='csv')
    r_hat = sampler.sample(rep,nChains=nChains,convergence_limit=convergence_limit, 
                           runs_after_convergence=runs_after_convergence)
    
    
    
    
    # Load the results gained with the dream sampler, stored in DREAM_hymod.csv
    results = spotpy.analyser.load_csv_results('DREAM_hymod')
    # Get fields with simulation data
    fields=[word for word in results.dtype.names if word.startswith('sim')]
    
    
    # Example plot to show remaining parameter uncertainty #
    fig= plt.figure(figsize=(16,9))
    ax = plt.subplot(1,1,1)
    q5,q25,q75,q95=[],[],[],[]
    for field in fields:
        q5.append(np.percentile(results[field][-100:-1],2.5))
        q95.append(np.percentile(results[field][-100:-1],97.5))
    ax.plot(q5,color='dimgrey',linestyle='solid')
    ax.plot(q95,color='dimgrey',linestyle='solid')
    ax.fill_between(np.arange(0,len(q5),1),list(q5),list(q95),facecolor='dimgrey',zorder=0,
                    linewidth=0,label='parameter uncertainty')  
    ax.plot(spot_setup.evaluation(),'r.',label='data')
    ax.set_ylim(-50,450)
    ax.set_xlim(0,729)
    ax.legend()
    fig.savefig('python_hymod.png',dpi=300)
    #########################################################
    
    
    # Example plot to show the convergence #################
    fig= plt.figure(figsize=(12,16))
    plt.subplot(2,1,1)
    for i in range(int(max(results['chain']))+1):
        index=np.where(results['chain']==i)
        plt.plot(results['like1'][index], label='Chain '+str(i+1))
    
    plt.ylabel('Likelihood value')
    plt.legend()
    
    ax =plt.subplot(2,1,2)
    r_hat=np.array(r_hat)
    ax.plot([1.2]*len(r_hat),'k--')
    for i in range(len(r_hat[0])):
        ax.plot(r_hat[:,i],label='x'+str(i+1))
    
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylim(-1,50)
    ax.set_ylabel('R$^d$ - convergence diagnostic')
    plt.xlabel('Number of chainruns')
    plt.legend()
    fig.savefig('python_hymod_convergence.png',dpi=300)
    ########################################################
    
    
    
    
    # Example plot to show the parameter distribution ######
    parameters = spotpy.parameter.get_parameters_array(spot_setup)
    
    
    min_vs,max_vs = parameters['minbound'], parameters['maxbound']
    
    fig= plt.figure(figsize=(16,16))
    plt.subplot(5,2,1)
    x = results['par'+str(parameters['name'][0])]
    for i in range(int(max(results['chain']))):
        index=np.where(results['chain']==i)
        plt.plot(x[index],'.')
    plt.ylabel('cmax')
    plt.ylim(min_vs[0],max_vs[0])
    
    
    plt.subplot(5,2,2)
    x = results['par'+parameters['name'][0]][int(len(results)*0.5):]
    normed_value = 1
    hist, bins = np.histogram(x, bins=20, density=True)
    widths = np.diff(bins)
    hist *= normed_value
    plt.bar(bins[:-1], hist, widths)
    plt.ylabel('cmax')
    plt.xlim(min_vs[0],max_vs[0])
    
    
    
    plt.subplot(5,2,3)
    x = results['par'+parameters['name'][1]]
    for i in range(int(max(results['chain']))):
        index=np.where(results['chain']==i)
        plt.plot(x[index],'.')
    plt.ylabel('bexp')
    plt.ylim(min_vs[1],max_vs[1])
    
    plt.subplot(5,2,4)
    x = results['par'+parameters['name'][1]][int(len(results)*0.5):]
    normed_value = 1
    hist, bins = np.histogram(x, bins=20, density=True)
    widths = np.diff(bins)
    hist *= normed_value
    plt.bar(bins[:-1], hist, widths)
    plt.ylabel('bexp')
    plt.xlim(min_vs[1],max_vs[1])
    
    
    
    plt.subplot(5,2,5)
    x = results['par'+parameters['name'][2]]
    for i in range(int(max(results['chain']))):
        index=np.where(results['chain']==i)
        plt.plot(x[index],'.')
    plt.ylabel('alpha')
    plt.ylim(min_vs[2],max_vs[2])
    
    
    plt.subplot(5,2,6)
    x = results['par'+parameters['name'][2]][int(len(results)*0.5):]
    normed_value = 1
    hist, bins = np.histogram(x, bins=20, density=True)
    widths = np.diff(bins)
    hist *= normed_value
    plt.bar(bins[:-1], hist, widths)
    plt.ylabel('alpha')
    plt.xlim(min_vs[2],max_vs[2])
    
    
    plt.subplot(5,2,7)
    x = results['par'+parameters['name'][3]]
    for i in range(int(max(results['chain']))):
        index=np.where(results['chain']==i)
        plt.plot(x[index],'.')
    plt.ylabel('Ks')
    plt.ylim(min_vs[3],max_vs[3])
    
    
    plt.subplot(5,2,8)
    x = results['par'+parameters['name'][3]][int(len(results)*0.5):]
    normed_value = 1
    hist, bins = np.histogram(x, bins=20, density=True)
    widths = np.diff(bins)
    hist *= normed_value
    plt.bar(bins[:-1], hist, widths)
    plt.ylabel('Ks')
    plt.xlim(min_vs[3],max_vs[3])
    
    
    plt.subplot(5,2,9)
    x = results['par'+parameters['name'][4]]
    for i in range(int(max(results['chain']))):
        index=np.where(results['chain']==i)
        plt.plot(x[index],'.')
    plt.ylabel('Kq')
    plt.ylim(min_vs[4],max_vs[4])
    plt.xlabel('Iterations')
    
    plt.subplot(5,2,10)
    x = results['par'+parameters['name'][4]][int(len(results)*0.5):]
    normed_value = 1
    hist, bins = np.histogram(x, bins=20, density=True)
    widths = np.diff(bins)
    hist *= normed_value
    plt.bar(bins[:-1], hist, widths)
    plt.ylabel('Kq')
    plt.xlabel('Parameter range')
    plt.xlim(min_vs[4],max_vs[4])
    plt.show()
    fig.savefig('python_parameters.png',dpi=300)
    ########################################################
    
