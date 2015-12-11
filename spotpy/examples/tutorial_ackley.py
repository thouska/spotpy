# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds the example code from the ackley tutorial web-documention.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import spotpy
from spotpy.examples.spot_setup_ackley import spot_setup


#Create samplers for every algorithm:
results=[]
spot_setup=spot_setup()
rep=5000

sampler=spotpy.algorithms.mc(spot_setup,    dbname='ackleyMC',    dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())
#
sampler=spotpy.algorithms.lhs(spot_setup,   dbname='ackleyLHS',   dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.mle(spot_setup,   dbname='ackleyMLE',   dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.mcmc(spot_setup,  dbname='ackleyMCMC',  dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.sceua(spot_setup, dbname='ackleySCEUA', dbformat='csv')
sampler.sample(rep,ngs=2)
results.append(sampler.getdata())

sampler=spotpy.algorithms.sa(spot_setup,    dbname='ackleySA',    dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.demcz(spot_setup, dbname='ackleyDEMCz', dbformat='csv')
sampler.sample(rep,nChains=30)
results.append(sampler.getdata())
#
sampler=spotpy.algorithms.rope(spot_setup,  dbname='ackleyROPE',  dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())



algorithms=['MC','LHS','MLE','MCMC','SCEUA','SA','DEMCz','ROPE']
results=[]
for algorithm in algorithms:
    results.append(spotpy.analyser.load_csv_results('ackley'+algorithm))


evaluation = spot_setup.evaluation()

spotpy.analyser.plot_objectivefunctiontraces(results,evaluation,algorithms)


