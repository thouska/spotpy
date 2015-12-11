# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds the example code from the Griewank tutorial web-documention.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import spotpy
from spotpy.examples.spot_setup_griewank import spot_setup


#Create samplers for every algorithm:
results=[]
spot_setup=spot_setup()
rep=5000

sampler=spotpy.algorithms.mc(spot_setup,    dbname='GriewankMC',    dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.lhs(spot_setup,   dbname='GriewankLHS',   dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.mle(spot_setup,   dbname='GriewankMLE',   dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.mcmc(spot_setup,  dbname='GriewankMCMC',  dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.sceua(spot_setup, dbname='GriewankSCEUA', dbformat='csv')
sampler.sample(rep,ngs=4)
results.append(sampler.getdata())

sampler=spotpy.algorithms.sa(spot_setup,    dbname='GriewankSA',    dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.demcz(spot_setup, dbname='GriewankDEMCz', dbformat='csv')
sampler.sample(rep,nChains=4)
results.append(sampler.getdata())

sampler=spotpy.algorithms.rope(spot_setup,  dbname='GriewankROPE',  dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())



algorithms=['MC','LHS','MLE','MCMC','SCEUA','SA','DEMCz','ROPE']
#results=[]
#for algorithm in algorithms:
#    results.append(spot.analyser.load_csv_results('Griewank'+algorithm))


evaluation = spot_setup.evaluation()

spotpy.analyser.plot_heatmap_griewank(results,algorithms)
