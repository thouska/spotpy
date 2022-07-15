"""
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Philipp Kraft
"""


class ForEach(object):
    def __init__(self, process):
        self.process = process
        self.phase = None

    def is_idle(self):
        return True

    def terminate(self):
        pass

    def setphase(self, phasename):
        self.phase = phasename

    def start(self):
        pass

    def __call__(self, jobs):
        for job in jobs:
            data = self.process(job)
            yield data
