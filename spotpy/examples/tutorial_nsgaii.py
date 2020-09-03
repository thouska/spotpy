#!/usr/bin/env python
# coding: utf-8


from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
    import spotpy
except ImportError:
    import sys

    sys.path.append(".")
    import spotpy

import spotpy.algorithms
import unittest
from spotpy.examples.spot_setup_hymod_python import spot_setup
import matplotlib.pyplot as plt
import numpy as np


def multi_obj_func(evaluation, simulation):
    #used to overwrite objective function in hymod example
    like1 = abs(spotpy.objectivefunctions.pbias(evaluation, simulation))
    like2 = spotpy.objectivefunctions.rmse(evaluation, simulation)
    like3 = spotpy.objectivefunctions.rsquared(evaluation, simulation)*-1
    return [like1, like2, like3]

if __name__ == "__main__":
    

    generations=40
    n_pop = 20
    skip_duplicates = True
    
    sp_setup=spot_setup(obj_func= multi_obj_func)
    sampler = spotpy.algorithms.NSGAII(spot_setup=sp_setup, dbname='NSGA2', dbformat="csv")
    
    sampler.sample(generations, n_obj= 3, n_pop = n_pop, skip_duplicates = skip_duplicates)
    #sampler.sample(generations=5, paramsamp=40)
    
    
#    # user config
#    
#    n_var = 5
#    
#    
#    last = None
#    first = None
#    
#    # output calibration 
#    
#    df = pd.read_csv("NSGA2.csv")
#    
#    if last:
#        df = df.iloc[-last:,:]
#    elif first:
#        df = df.iloc[:first,:]
#    else:
#        pass
#    
#    
#    
#    print(len(df))
#    # plot objective functions
#    fig = plt.figure()
#    for i,name in enumerate(df.columns[:n_obj]):
#        ax = fig.add_subplot(n_obj,1,i +1)
#        df.loc[::5,name].plot(lw=0.5,figsize=(18,8),ax = ax,color="black")
#        plt.title(name)
#    plt.show()
#    
#    last = 100
#    first = None
#    
#    x,y,z = df.iloc[-last:,0],df.iloc[-last:,1],df.iloc[-last:,2]
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(x,y,z,marker="o")
#    ax.set_xlabel("f0")
#    ax.set_ylabel("f1")
#    ax.set_zlabel("f2")
#    plt.show()
#    
#    # plot parameters
#    fig = plt.figure()
#    for i,name in enumerate(df.columns[n_obj:8]):
#        ax = fig.add_subplot(5,1,i +1)
#        df.loc[:,name].plot(lw=0.5,figsize=(18,8),ax = ax,color="black")
#        plt.title(name)
#    plt.show()
#    



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
    ax11.plot(sp_setup.evaluation(),'r.',label='data')
    ax11.set_ylim(-50,450)
    ax11.set_xlim(0,729)
    ax11.legend()
    fig.savefig('python_hymod.png',dpi=300)
    #########################################################


    x,y,z = results['like1'][-n_pop:],results['like2'][-n_pop:],results['like3'][-n_pop:]
    fig = plt.figure(4)
    ax12 = fig.add_subplot(111, projection='3d')
    ax12.scatter(x,y,z,marker="o")
    ax12.set_xlabel("pbias")
    ax12.set_ylabel("rmse")
    ax12.set_zlabel("rsquared")
    plt.show()


    
class Test_NSGAII(unittest.TestCase):
    def multi_obj_func(evaluation, simulation):
        #used to overwrite objective function in hymod example
        like1 = abs(spotpy.objectivefunctions.pbias(evaluation, simulation))
        like2 = spotpy.objectivefunctions.rmse(evaluation, simulation)
        like3 = spotpy.objectivefunctions.rsquared(evaluation, simulation)*-1
        return [like1, like2, like3]

    def setUp(self):
        generations=40
        n_pop = 20
        skip_duplicates = True

        self.sp_setup=spot_setup(obj_func= multi_obj_func)
        self.sampler = spotpy.algorithms.NSGAII(self.sp_setup, dbname='NSGA2', dbformat="csv")
        self.sampler.sample(generations, n_obj= 3, n_pop = n_pop, skip_duplicates = skip_duplicates)

    def test_sampler_output(self):
        self.assertEqual(200, len(self.sampler.getdata()))
