from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from pprint import pprint
import numpy as np
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

# Create samplers for every algorithm:
results = []
benchmarks_duration = {"dds":[], "sceua":[], "dds_like":[],"sceua_like":[]}
reps = [300, 1000, 3000, 4000, 5000, 10000]


for rep in reps:

    timeout = 10  # Given in Seconds

    parallel = "mpc"
    dbformat = "csv"

    start = time.time()
    dds_sampler = spotpy.algorithms.DDS(spot_setup, parallel=parallel, dbname='DDS', dbformat=dbformat, sim_timeout=timeout)
    dds_sampler.sample(rep, trials=1)
    results.append(dds_sampler.getdata())
    dds_elapsed = time.time() - start

    benchmarks_duration["dds"].append(dds_elapsed)
    benchmarks_duration["dds_like"].append(dds_sampler.status.objectivefunction)




def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')





fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
rep_labels = [str(j) for j in reps]
x_pos = [i for i, _ in enumerate(rep_labels)]
X = np.arange(len(benchmarks_duration["dds"]))
dds_time = ax.bar(x_pos, benchmarks_duration["dds"], color = 'b', width = 0.45)

plt.xticks(x_pos, rep_labels)
plt.xlabel("Repetitions")
plt.ylabel("Best Objective Function Value")

autolabel(dds_plot)
autolabel(sceua_plot)

plt.savefig("MPI_TEST")


dds_like = ax.bar([j+0.45 for j in x_pos], benchmarks_duration["dds_like"], color = 'g', width = 0.45)

plt.ylabel("Best Objective Function Value")