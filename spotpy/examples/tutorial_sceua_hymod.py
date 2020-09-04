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
    rep=1000
    filename = 'SCEUA_hymod'
    # Create the SCE-UA sampler of spotpy, alt_objfun is set to None to force SPOTPY
    # to jump into the def objectivefunction in the spot_setup class (default is
    # spotpy.objectivefunctions.rmse) 
    sampler=spotpy.algorithms.sceua(spot_setup, dbname='SCEUA_hymod', dbformat='csv')
    
    #Start the sampler, one can specify ngs, kstop, peps and pcento id desired
    sampler.sample(rep, ngs=7, kstop=3, peps=0.1, pcento=0.1) 
    
    # Load the results gained with the sceua sampler, stored in SCEUA_hymod.csv
    results = spotpy.analyser.load_csv_results('SCEUA_hymod')
    

    print(len(results), 'runs were saved.')
    
    fig= plt.figure(1,figsize=(9,5))
    plt.plot(results['like1'])
    plt.show()
    plt.ylabel('RMSE')
    plt.xlabel('Iteration')
    fig.savefig('hymod_objectivefunction.png',dpi=300)
    
    # Example plot to show the parameter distribution ###### 
    fig= plt.figure(2,figsize=(9,9))
    normed_value = 1
    
    plt.subplot(5,2,1)
    x = results['parcmax']
    for i in range(int(max(results['chain'])-1)):
        index=np.where(results['chain']==i+1) #Ignores burn-in chain
        plt.plot(x[index],'.')
    plt.ylabel('cmax')
    plt.ylim(spot_setup.cmax.minbound, spot_setup.cmax.maxbound)
    
    
    plt.subplot(5,2,2)
    x = x[int(len(results)*0.9):] #choose the last 10% of the sample
    hist, bins = np.histogram(x, bins=20, density=True)
    widths = np.diff(bins)
    hist *= normed_value
    plt.bar(bins[:-1], hist, widths)
    plt.ylabel('cmax')
    plt.xlim(spot_setup.cmax.minbound, spot_setup.cmax.maxbound)
    
    
    plt.subplot(5,2,3)
    x = results['parbexp']
    for i in range(int(max(results['chain'])-1)):
        index=np.where(results['chain']==i+1)
        plt.plot(x[index],'.')
    plt.ylabel('bexp')
    plt.ylim(spot_setup.bexp.minbound, spot_setup.bexp.maxbound)
    
    plt.subplot(5,2,4)
    x = x[int(len(results)*0.9):]
    hist, bins = np.histogram(x, bins=20, density=True)
    widths = np.diff(bins)
    hist *= normed_value
    plt.bar(bins[:-1], hist, widths)
    plt.ylabel('bexp')
    plt.xlim(spot_setup.bexp.minbound, spot_setup.bexp.maxbound)
    
    
    
    plt.subplot(5,2,5)
    x = results['paralpha']
    for i in range(int(max(results['chain'])-1)):
        index=np.where(results['chain']==i+1)
        plt.plot(x[index],'.')
    plt.ylabel('alpha')
    plt.ylim(spot_setup.alpha.minbound, spot_setup.alpha.maxbound)
    
    
    plt.subplot(5,2,6)
    x = x[int(len(results)*0.9):]
    hist, bins = np.histogram(x, bins=20, density=True)
    widths = np.diff(bins)
    hist *= normed_value
    plt.bar(bins[:-1], hist, widths)
    plt.ylabel('alpha')
    plt.xlim(spot_setup.alpha.minbound, spot_setup.alpha.maxbound)
    
    
    plt.subplot(5,2,7)
    x = results['parKs']
    for i in range(int(max(results['chain'])-1)):
        index=np.where(results['chain']==i+1)
        plt.plot(x[index],'.')
    plt.ylabel('Ks')
    plt.ylim(spot_setup.Ks.minbound, spot_setup.Ks.maxbound)
    
    
    plt.subplot(5,2,8)
    x = x[int(len(results)*0.9):]

    hist, bins = np.histogram(x, bins=20, density=True)
    widths = np.diff(bins)
    hist *= normed_value
    plt.bar(bins[:-1], hist, widths)
    plt.ylabel('Ks')
    plt.xlim(spot_setup.Ks.minbound, spot_setup.Ks.maxbound)
    
    
    plt.subplot(5,2,9)
    x = results['parKq']
    for i in range(int(max(results['chain'])-1)):
        index=np.where(results['chain']==i+1)
        plt.plot(x[index],'.')
    plt.ylabel('Kq')
    plt.ylim(spot_setup.Kq.minbound, spot_setup.Kq.maxbound)
    plt.xlabel('Iterations')
    
    plt.subplot(5,2,10)
    x = x[int(len(results)*0.9):]
    hist, bins = np.histogram(x, bins=20, density=True)
    widths = np.diff(bins)
    hist *= normed_value
    plt.bar(bins[:-1], hist, widths)
    plt.ylabel('Kq')
    plt.xlabel('Parameter range')
    plt.xlim(spot_setup.Kq.minbound, spot_setup.Kq.maxbound)
    plt.show()
    fig.savefig('hymod_parameters.png',dpi=300)