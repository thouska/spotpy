'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Philipp Kraft
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pathos.multiprocessing as mp

process_count = None

class PhaseChange(object):
    """
    Object to identify a change of a simulation phase
    """
    def __init__(self, phase):
        self.phase = phase
        
class ForEach(object):
    """
    ForEach is a classes for multiprocessed work based on a generater object which is given if __call__ is called
    We using the pathos multiprocessing module and the orderd map function where results are saved until results in
    the given order are caluculated. We yielding back the result so a generator object is created.
    """
    def __init__(self, process):
        self.size = process_count or mp.cpu_count()
        self.process = process
        self.phase=None
        self.pool = mp.ProcessingPool(self.size)

    def is_idle(self):
        return False

    def terminate(self):
        pass

    def start(self):
        pass

    def setphase(self, phasename):
        self.phase = phasename

    def f(self, job):
        data = self.process(job)
        return data

    def __call__(self, jobs):
        results = self.pool.map(self.f, jobs)
        for i in results:
            yield i
