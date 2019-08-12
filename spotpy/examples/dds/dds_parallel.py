from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import sys
import os
import matplotlib.pylab as plt
import json
import time

try:
    import spotpy
except ImportError:
    import sys

    sys.path.append(".")
    import spotpy

from spotpy.examples.spot_setup_hymod_python import spot_setup

spot_setup = spot_setup()


path = os.path.abspath(os.path.dirname(__file__))
json_path = path + "/dds_parallel_data.json"
benchmarks_duration = json.load(open(json_path))


rep = int(sys.argv[1])
timeout = 10  # Given in Seconds
parallel = "mpi"
dbformat = "csv"
start = time.time()
dds_sampler = spotpy.algorithms.dds(spot_setup, parallel=parallel, dbname='DDS', dbformat=dbformat, sim_timeout=timeout)
dds_sampler.sample(rep, trials=1)
dds_elapsed = time.time() - start
print(dds_elapsed)

benchmarks_duration["dds_duration"].append(dds_elapsed)
benchmarks_duration["dds_like"].append(dds_sampler.status.objectivefunction_max)
benchmarks_duration["rep"].append(rep)

print(benchmarks_duration)

json.dump(benchmarks_duration, open(json_path,"w"))