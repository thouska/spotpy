#!/usr/bin/env python
# coding: utf-8

"""

"""

from __future__ import division, print_function, unicode_literals
import sys
import datetime

import spotpy
import os

from spotpy.examples.spot_setup_cmf_lumped import SingleStorage

def parallel():
    """
    Returns 'mpi', if this code runs with MPI, else returns 'seq'
    :return:
    """
    return 'mpi' if 'OMPI_COMM_WORLD_SIZE' in os.environ else 'seq'


def get_runs(default=1):
    """
    Returns the number of runs, given by commandline or variable
    :param default: Return this if no other source for number of runs has been found
    :return: int
    """
    # Get number of runs
    if 'SPOTPYRUNS' in os.environ:
        # from environment
        return int(os.environ['SPOTPYRUNS'])
    elif len(sys.argv) > 1:
        # from command line
        return int(sys.argv[1])
    else:
        # run once
        return default


if __name__ == '__main__':

    # Create the model
    model = SingleStorage(datetime.datetime(1980, 1, 1),
                          datetime.datetime(1985, 12, 31))

    runs = get_runs(default=5)
    # Create the sampler
    sampler = spotpy.algorithms.mc(model,
                                   parallel=parallel(),
                                   dbname=model.dbname, dbformat='csv',
                                   save_sim=True, sim_timeout=300)

    # Now we can sample with the implemented Monte Carlo algorithm:
    sampler.sample(runs)



