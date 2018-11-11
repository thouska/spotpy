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
rep=3000
timeout=10 #Given in Seconds


parallel = "seq"
dbformat = "csv"
sampler=spotpy.algorithms.DDS(spot_setup,parallel=parallel, dbname='DDS', dbformat=dbformat, sim_timeout=timeout)
print(describe(sampler))
sampler.sample(rep, trials=1, r=0.1)
results.append(sampler.getdata())



sampler=spotpy.algorithms.dream(spot_setup,parallel=parallel, dbname='DDS', dbformat=dbformat, sim_timeout=timeout)
print(describe(sampler))
sampler.sample(rep)
results.append(sampler.getdata())



#algorithms = ['mc','lhs','mle','mcmc','sceua','sa','demcz','rope','abc','fscabc', 'demcz', 'dream']
spotpy.analyser.plot_parametertrace_algorithms(results,algorithmnames=["DDS","DREAM"],parameternames=['0','1'])


print(results[0].dtype) # Check for Travis: Get the last sampled parameter for x
#evaluation = spot_setup.evaluation()
