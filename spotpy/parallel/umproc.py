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

class PhaseChange(object):
    """
    Object to identify a change of a simulation phase
    """
    def __init__(self,phase):
        self.phase=phase
        
class ForEach(object):
    def __init__(self,process):
        self.size = mp.cpu_count()
        self.process = process
        self.phase=None
        self.pool = mp.ProcessingPool(self.size)

    def is_idle(self):
        return False
    def terminate(self):
        pass

    def start(self):
        pass
    def setphase(self,phasename):
        self.phase=phasename


    def f(self, job):
        data = self.process(job)
        #print(data)
        return data

    def __call__(self,jobs):
        results = self.pool.uimap(self.f, jobs)
        for i in results:
            yield i


