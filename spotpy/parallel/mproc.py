'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Philipp Kraft
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import multiprocessing as mp

class ForEach(object):
    def __init__(self,process):
        self.size = mp.cpu_count()
        self.pool = mp.Pool()
        self.process = process
    def is_idle(self):
        return False
    def terminate(self):
        self.pool.join()
    def __call__(self,jobs):
        for result in self.pool.imap_unordered(self.process, jobs):
            yield result
def proc(j):
    for i in xrange(10000):
        q = i,i ** 2
    return j,j**2
if __name__ == '__main__':
    fe = ForEach(proc)
    jobs = range(10000)
    for j,q in fe(jobs):
        print(j)
