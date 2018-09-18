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


from spotpy.describe import describe

from spotpy.examples.spot_setup_dds import spot_setup
from spotpy.examples.spot_setup_dds import ackley10

#Create samplers for every algorithm:
results=[]
spot_setup=spot_setup()
rep=1000
timeout=10 #Given in Seconds



Initial_solution = [] # TODO if user had seom, read it in




parallel = "seq"
dbformat = "csv"

sampler=spotpy.algorithms.DDS(spot_setup,parallel=parallel, dbname='DDS', dbformat=dbformat, sim_timeout=timeout)

print(describe(sampler))
sampler.sample(rep,fraction1=0.2,trials=2)

#print(sampler.getdata())

results.append(sampler.getdata())



#print(results[0].dtype) # Check for Travis: Get the last sampled parameter for x
evaluation = spot_setup.evaluation()
