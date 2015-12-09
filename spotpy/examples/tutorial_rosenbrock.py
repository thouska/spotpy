# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This class holds the example code from the Rosenbrock tutorial web-documention.
'''

import spotpy
from spot_setup_ackley import spot_setup
from spotpy import analyser

#Create samplers for every algorithm:
results=[]
spot_setup=spot_setup()
rep=5000

sampler=spotpy.algorithms.mc(spot_setup,    dbname='RosenMC',    dbformat='csv')
results.append(sampler.sample(rep))

sampler=spotpy.algorithms.lhs(spot_setup,   dbname='RosenLHS',   dbformat='csv')
results.append(sampler.sample(rep))

sampler=spotpy.algorithms.mle(spot_setup,   dbname='RosenMLE',   dbformat='csv')
results.append(sampler.sample(rep))

sampler=spotpy.algorithms.mcmc(spot_setup,  dbname='RosenMCMC',  dbformat='csv')
results.append(sampler.sample(rep))

sampler=spotpy.algorithms.sceua(spot_setup, dbname='RosenSCEUA', dbformat='csv')
results.append(sampler.sample(rep,ngs=4))

sampler=spotpy.algorithms.sa(spot_setup,    dbname='RosenSA',    dbformat='csv')
results.append(sampler.sample(rep))

sampler=spotpy.algorithms.demcz(spot_setup, dbname='RosenDEMCz', dbformat='csv')
results.append(sampler.sample(rep,nChains=4))

sampler=spotpy.algorithms.rope(spot_setup,  dbname='RosenROPE',  dbformat='csv')
results.append(sampler.sample(rep))




algorithms=['MC','LHS','MLE','MCMC','SCEUA','SA','DEMCz','ROPE']
results=[]
for algorithm in algorithms:
    results.append(spotpy.analyser.load_csv_results('Rosen'+algorithm))


evaluation = spot_setup.evaluation()

spotpy.analyser.plot_parametertrace_algorithms(results,algorithmnames=algorithms,parameternames=['x','y']) 