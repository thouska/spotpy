# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Philipp Kraft

This class makes the MPI parallelization.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from mpi4py import MPI


class tag:
    job=11
    answer=12
class PhaseChange(object):
    """
    Object to identify a change of a simulation phase
    """
    def __init__(self,phase):
        self.phase=phase
class ForEach(object):
    """
    This is the mpi version of the spot repetition object.
    Repitition objects are owned by spot algorithms and can be used to repeat a task
    in a for loop. Using the repetition instead of a normal iterable allows for parallelization,
    in this case using mpi
    """
    def __repr__(self):
        text="ForEach(rank=%i/%i,phase=%s)" % (self.rank,self.size,self.phase)
        return(text)
    def __init__(self,process):
        """
        Creates a repetition around a callable
        """
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        if self.size<=1:
            raise RuntimeError('Need at least two processes for parallelization')
        self.rank = self.comm.Get_rank()
        self.process = process
        self.phase=None
        if self.rank == 0:
            self.slots = [None] * (self.size-1)
    def is_idle(self):
        if self.rank == 0:
            return all(s is None for s in self.slots)
        else:
            return False
    def terminate(self):
        if self.rank == 0:
            for i in range(1,self.size):
                self.comm.send(StopIteration(),dest=i,tag=tag.job)
        else:
            raise RuntimeError("Don't call terminate on worker")
    def setphase(self,phasename):
        if self.rank == 0:
            for i in range(1,self.size):
                self.comm.send(PhaseChange(phasename),dest=i,tag=tag.job)
            self.phase=phasename
        else:
            raise RuntimeError("Don't call setphase on worker")
    def __recv(self):
        while True and self.rank>0:
            obj = self.comm.recv(source=0, tag=tag.job)
            if type(obj) is StopIteration:
                break
            elif type(obj) is PhaseChange:
                self.phase = obj.phase
            else: # obj is a job for self.process
                yield obj
    
    def __foreach(self):
        assert(self.rank>0)
        for job in self.__recv():
            res = self.process(job)
            self.__send(res)
        exit()
    def __send(self,arg):
        if self.rank == 0:
            for i,slot in enumerate(self.slots):
                # found a free slot
                if slot is None:
                    # Move job from queue to job
                    try:
                        job = arg.next()
                        self.slots[i] = job
                        # Send slot job to dest
                        self.comm.send(job, dest=i+1, tag=tag.job)
                    except StopIteration:
                        return False
            return True
        else:
            self.comm.send(arg,dest=0,tag=tag.answer)
            return True
    def start(self):
        if self.rank:
            self.__foreach()

    def __call__(self,jobs):
        jobiter = iter(jobs)
        while self.__send(jobiter) or not self.is_idle():
            for i,slot in enumerate(self.slots):
                if not slot is None: # If slot is active
                    # Check if slot is ready
                    if self.comm.Iprobe(source=i+1,tag=tag.answer):
                        # Receive data
                        data = self.comm.recv(source=i+1,tag=tag.answer)
                        # Free slot
                        self.slots[i] = None
                        yield data
        
