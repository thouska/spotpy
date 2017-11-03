# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds example code how to use the dream algorithm
'''

import numpy as np
import spotpy
from spotpy.examples.spot_setup_hymod import spot_setup
import pylab as plt


# Initialize the Hymod example (will only work on Windows systems)
spot_setup=spot_setup()

# Create the Dream sampler of spotpy, al_objfun is set to None to force SPOTPY
# to jump into the def objectivefunction in the spot_setup class (default is
# spotpy.objectivefunctions.log_p) 
sampler=spotpy.algorithms.dream(spot_setup, dbname='DREAM_hymod', dbformat='csv',
                                alt_objfun=None)

#Select number of maximum repetitions
rep=10000

# Select five chains and set the Gelman-Rubin convergence limit
nChains                = 4
convergence_limit      = 1.2
runs_after_convergence = 100

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
ax.fill_between(np.arange(0,730,1),list(q5),list(q95),facecolor='dimgrey',zorder=0,
                linewidth=0,label='parameter uncertainty')  
ax.plot(spot_setup.evaluation(),'r.',label='data')
ax.set_ylim(-50,450)
ax.set_xlim(0,729)
ax.legend()
fig.savefig('python_hymod.png',dpi=300)
#########################################################


# Example plot to show the convergence #################
fig= plt.figure(figsize=(16,9))
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

def find_min_max(spot_setup):
    randompar=spot_setup.parameters()['random']        
    for i in range(1000):
        randompar=np.column_stack((randompar,spot_setup.parameters()['random']))
    return np.amin(randompar,axis=1),np.amax(randompar,axis=1)


min_vs,max_vs = find_min_max(spot_setup)

fig= plt.figure(figsize=(16,16))
plt.subplot(5,2,1)
x = results['par'+str(spot_setup.parameters()['name'][0].decode())]
for i in range(int(max(results['chain']))):
    index=np.where(results['chain']==i)
    plt.plot(x[index],'.')
plt.ylabel('x1')
plt.ylim(min_vs[0],max_vs[0])


plt.subplot(5,2,2)
x = results['par'+spot_setup.parameters()['name'][0].decode()][int(len(results)*0.5):]
normed_value = 1
hist, bins = np.histogram(x, bins=20, density=True)
widths = np.diff(bins)
hist *= normed_value
plt.bar(bins[:-1], hist, widths)
plt.ylabel('x1')
plt.xlim(min_vs[0],max_vs[0])



plt.subplot(5,2,3)
x = results['par'+spot_setup.parameters()['name'][1].decode()]
for i in range(int(max(results['chain']))):
    index=np.where(results['chain']==i)
    plt.plot(x[index],'.')
plt.ylabel('x2')
plt.ylim(min_vs[1],max_vs[1])

plt.subplot(5,2,4)
x = results['par'+spot_setup.parameters()['name'][1].decode()][int(len(results)*0.5):]
normed_value = 1
hist, bins = np.histogram(x, bins=20, density=True)
widths = np.diff(bins)
hist *= normed_value
plt.bar(bins[:-1], hist, widths)
plt.ylabel('x2')
plt.xlim(min_vs[1],max_vs[1])



plt.subplot(5,2,5)
x = results['par'+spot_setup.parameters()['name'][2].decode()]
for i in range(int(max(results['chain']))):
    index=np.where(results['chain']==i)
    plt.plot(x[index],'.')
plt.ylabel('x3')
plt.ylim(min_vs[2],max_vs[2])


plt.subplot(5,2,6)
x = results['par'+spot_setup.parameters()['name'][2].decode()][int(len(results)*0.5):]
normed_value = 1
hist, bins = np.histogram(x, bins=20, density=True)
widths = np.diff(bins)
hist *= normed_value
plt.bar(bins[:-1], hist, widths)
plt.ylabel('x3')
plt.xlim(min_vs[2],max_vs[2])


plt.subplot(5,2,7)
x = results['par'+spot_setup.parameters()['name'][3].decode()]
for i in range(int(max(results['chain']))):
    index=np.where(results['chain']==i)
    plt.plot(x[index],'.')
plt.ylabel('x4')
plt.ylim(min_vs[3],max_vs[3])


plt.subplot(5,2,8)
x = results['par'+spot_setup.parameters()['name'][3].decode()][int(len(results)*0.5):]
normed_value = 1
hist, bins = np.histogram(x, bins=20, density=True)
widths = np.diff(bins)
hist *= normed_value
plt.bar(bins[:-1], hist, widths)
plt.ylabel('x4')
plt.xlim(min_vs[3],max_vs[3])


plt.subplot(5,2,9)
x = results['par'+spot_setup.parameters()['name'][4].decode()]
for i in range(int(max(results['chain']))):
    index=np.where(results['chain']==i)
    plt.plot(x[index],'.')
plt.ylabel('x5')
plt.ylim(min_vs[4],max_vs[4])
plt.xlabel('Iterations')

plt.subplot(5,2,10)
x = results['par'+spot_setup.parameters()['name'][4].decode()][int(len(results)*0.5):]
normed_value = 1
hist, bins = np.histogram(x, bins=20, density=True)
widths = np.diff(bins)
hist *= normed_value
plt.bar(bins[:-1], hist, widths)
plt.ylabel('x5')
plt.xlabel('Parameter range')
plt.xlim(min_vs[4],max_vs[4])
plt.show()
fig.savefig('python_parameters.png',dpi=300)
########################################################

