# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska
This class holds the example code from the Rosenbrock tutorial web-documention.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import spotpy
from spotpy.examples.spot_setup_rosenbrock import spot_setup

#Create samplers for every algorithm:
results=[]
spot_setup=spot_setup()
rep=5000

sampler=spotpy.algorithms.mc(spot_setup,    dbname='RosenMC',    dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.lhs(spot_setup,   dbname='RosenLHS',    dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.mle(spot_setup,   dbname='RosenMLE',   dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.mcmc(spot_setup,  dbname='RosenMCMC',  dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.sceua(spot_setup, dbname='RosenSCEUA', dbformat='csv')
sampler.sample(rep,ngs=4)
results.append(sampler.getdata())

sampler=spotpy.algorithms.sa(spot_setup,    dbname='RosenSA',    dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.demcz(spot_setup, dbname='RosenDEMCz',  dbformat='csv')
sampler.sample(rep,nChains=4)
results.append(sampler.getdata())

sampler=spotpy.algorithms.rope(spot_setup,  dbname='RosenROPE',  dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.abc(spot_setup,    dbname='RosenMC',     dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.fscabc(spot_setup,    dbname='RosenMC',  dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.dream(spot_setup,    dbname='RosenMC',  dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

print(results[0].dtype) # Check for Travis: Get the last sampled parameter for x
evaluation = spot_setup.evaluation()

#Example how to plot the data
#algorithms = ['mc','lhs','mle','mcmc','sceua','sa','demcz','rope','abc','fscabc','dream']
#spotpy.analyser.plot_parametertrace_algorithms(results,algorithmnames=algorithms,parameternames=['x','y']) 
