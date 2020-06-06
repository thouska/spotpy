from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import spotpy


from spotpy.examples.spot_setup_dtlz1 import spot_setup


#Create samplers for every algorithm:
results=[]
spot_setup=spot_setup(n_var=5,n_obj=3)
generations=800
paramsamp = 30

sampler=spotpy.algorithms.NSGAII(spot_setup,    dbname='NSGA2',    dbformat='csv',save_sim=True)
sampler.sample(generations=generations, paramsamp=paramsamp)
results.append(sampler.getdata())

print(results)
