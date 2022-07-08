# -*- coding: utf-8 -*-
"""
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Philipp Kraft

This class makes the MPI parallelization.

When an algorithm is constructed with parallel='mpi' the repeater of the algorithm as
a ForEach object from this package. The .start() method seperates one master process from
all the other processes that are used as workers. The master holds a list called "slots",
where the master notes which processes are busy with a job. When a new job should be sent to
a worker, the master looks for a free slot and sends the job to the corresponding worker
process.
"""

from mpi4py import MPI


class tag:
    """
    This is just an enum to identify messages
    """

    job = 11
    answer = 12


class PhaseChange(object):
    """
    Object to identify a change of a simulation phase
    """

    def __init__(self, phase):
        self.phase = phase


class ForEach(object):
    """
    This is the mpi version of the spot repetition object.
    Repitition objects are owned by spot algorithms and can be used to repeat a task
    in a for loop. Using the repetition instead of a normal iterable allows for parallelization,
    in this case using mpi

    Attributes:
        size: number of mpi processes
        rank: current process number
        process: callable to execute
        phase: The phase of the job (used by process)
        on_worker_terminate: An optional callable, that gets executed
                             when the worker processes terminate
    """

    def __repr__(self):
        return "ForEach(rank=%i/%i,phase=%s)" % (self.rank, self.size, self.phase)

    def __init__(self, process):
        """
        Creates a repetition around a callable

        :param process: A callable to process the data
        """
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        if self.size <= 1:
            raise RuntimeError("Need at least two processes for parallelization")
        self.rank = self.comm.Get_rank()
        self.process = process
        self.phase = None
        self.on_worker_terminate = None
        if self.rank == 0:
            # The slots are a place for the master to remember which worker is doing something
            # Idle slots contain None
            self.slots = [None] * (self.size - 1)

    def is_master(self):
        """
        :return: True if this Repititor lives on the master process
        """
        return self.rank == 0

    def is_worker(self):
        """

        :return: True if self lives on a worker process
        """
        return self.rank > 0

    def is_idle(self):
        """

        :return: True, if all slots are empty
        """
        assert self.is_master()
        return all(s is None for s in self.slots)

    def terminate(self):
        """
        Sends a termination command to all workers
        :return:
        """
        assert self.is_master(), "Don't call terminate on worker"
        for i in range(1, self.size):
            self.comm.send(StopIteration(), dest=i, tag=tag.job)

    def setphase(self, phasename):
        """
        Sends out to all workers that a new phase has come
        :param phasename:
        :return:
        """
        assert self.is_master()
        for i in range(1, self.size):
            self.comm.send(PhaseChange(phasename), dest=i, tag=tag.job)
        self.phase = phasename

    def __wait(self):
        """
        The loop where a worker is waiting for jobs.
        Breaks when master sent StopIteration
        :return: Nothing, calls exit()
        """
        try:
            assert self.is_worker()
            while True:
                # Wait for a message
                obj = self.comm.recv(source=0, tag=tag.job)
                # Handle messages
                if type(obj) is StopIteration:
                    # Stop message
                    break
                elif type(obj) is PhaseChange:
                    # Phase change
                    self.phase = obj.phase
                else:  # obj is a job for self.process
                    # Send the object back for processing it
                    res = self.process(obj)
                    self.comm.send(res, dest=0, tag=tag.answer)

            if callable(self.on_worker_terminate):
                self.on_worker_terminate()

        finally:
            exit()

    def __send(self, jobiter):
        """
        The master uses this function to send jobs to the workers
        First it looks for a free slots, and then the jobs go there
        Used by __call__
        :param jobiter: An iterator over job arguments
        :return: True if there are pending jobs
        """
        assert self.is_master()
        for i, slot in enumerate(self.slots):
            # found a free slot
            if slot is None:
                # Move job from queue to job
                try:
                    # Changed from arg.next() which is not really py3 compliant
                    job = next(jobiter)
                    self.slots[i] = job
                    # Send slot job to destination rank
                    self.comm.send(job, dest=i + 1, tag=tag.job)
                except StopIteration:
                    return False
        return True

    def start(self):
        """
        Sepearates the master from the workers
        Sends all workers into wait modus, the master will just proceed
        :return:
        """
        if self.is_worker():
            self.__wait()

    def __call__(self, jobs):
        """
        Sends the jobs out to the workers and receives the results
        :param jobs: an iterable of jobs to be sent to the workers. Each job contains
                    the args of the process function
        :return: Yields the received result
        """

        # Get the iterator for the iterable
        jobiter = iter(jobs)
        # Send out job while we have them
        # __send(jobiter) returns False when all jobs are send to the workers
        while self.__send(jobiter) or not self.is_idle():
            # Loop over the slots to get results
            for i, slot in enumerate(self.slots):
                # If slot is active
                if slot is not None:
                    # Check if slot has data in the pipeline
                    if self.comm.Iprobe(source=i + 1, tag=tag.answer):
                        # Receive data
                        data = self.comm.recv(source=i + 1, tag=tag.answer)
                        # Free slot
                        self.slots[i] = None
                        yield data
