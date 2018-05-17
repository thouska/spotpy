# -*- coding: utf-8 -*-
'''
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska and the SALib team
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import _algorithm
import numpy as np
import math


class fast(_algorithm):
    '''
    Fourier Amplitude Sensitivity Test (FAST)
    
    This class holds the Fourier Amplitude Sensitivity Test (FAST) based on Cukier et al. (1973) and Saltelli et al. (1999):

    Cukier, R. I., Fortuin, C. M., Shuler, K. E., Petschek, A. G. and Schaibly, J. H.: Study of the sensitivity of coupled reaction systems to uncertainties in rate coefficients. I Theory, J. Chem. Phys., 59(8), 3873–3878, 1973.
    
    Saltelli, A., Tarantola, S. and Chan, K. P.-S.: A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output, Technometrics, 41(1), 39–56, doi:10.1080/00401706.1999.10485594, 1999.

    The presented code is based on SALib
    Copyright (C) 2013-2015 Jon Herman and others. Licensed under the GNU Lesser General Public License.
    The Sensitivity Analysis Library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
    The Sensitivity Analysis Library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License along with the Sensitivity Analysis Library. If not, see http://www.gnu.org/licenses/.
     '''

    def __init__(self,  *args, **kwargs):
        '''
        Input
        ----------
        spot_setup: class
            model: function 
                Should be callable with a parameter combination of the parameter-function 
                and return an list of simulation results (as long as evaluation list)
            parameter: function
                When called, it should return a random parameter combination. Which can 
                be e.g. uniform or Gaussian
            objectivefunction: function 
                Should return the objectivefunction for a given list of a model simulation and 
                observation.
            evaluation: function
                Should return the true values as return by the model.
    
        dbname: str
            * Name of the database where parameter, objectivefunction value and simulation results will be saved.
    
        dbformat: str
            * ram: fast suited for short sampling time. no file will be created and results are saved in an array.
            * csv: A csv file will be created, which you can import afterwards.        
    
        parallel: str
            * seq: Sequentiel sampling (default): Normal iterations on one core of your cpu.
            * mpi: Message Passing Interface: Parallel computing on cluster pcs (recommended for unix os).
    
        save_sim: boolean
            *True:  Simulation results will be saved
            *False: Simulationt results will not be saved
        '''
        super(fast, self).__init__(*args, **kwargs)
#        _algorithm.__init__(self, spot_setup, dbname=dbname,
#                            dbformat=dbformat, parallel=parallel, save_sim=save_sim,
#                           save_threshold=save_threshold)

    def scale_samples(self, params, bounds):
        '''
        Rescales samples in 0-to-1 range to arbitrary bounds.

        Arguments:
            bounds - list of lists of dimensions num_params-by-2
            params - numpy array of dimensions num_params-by-N,
            where N is the number of samples
        '''
        # Check bounds are legal (upper bound is greater than lower bound)
        b = np.array(bounds)
        lower_bounds = b[:, 0]
        upper_bounds = b[:, 1]

        if np.any(lower_bounds >= upper_bounds):
            raise ValueError("Bounds are not legal")

        # This scales the samples in-place, by using the optional output
        # argument for the numpy ufunctions
        # The calculation is equivalent to:
        #   sample * (upper_bound - lower_bound) + lower_bound
        np.add(np.multiply(params,
                           (upper_bounds - lower_bounds),
                           out=params),
               lower_bounds,
               out=params)

    def matrix(self, bounds, N, M=4):
        D = len(bounds)

        omega = np.empty([D])
        omega[0] = math.floor((N - 1) / (2 * M))
        m = math.floor(omega[0] / (2 * M))

        if m >= (D - 1):
            omega[1:] = np.floor(np.linspace(1, m, D - 1))
        else:
            omega[1:] = np.arange(D - 1) % m + 1

        # Discretization of the frequency space, s
        s = (2 * math.pi / N) * np.arange(N)

        # Transformation to get points in the X space
        X = np.empty([N * D, D])
        omega2 = np.empty([D])

        for i in range(D):
            omega2[i] = omega[0]
            idx = list(range(i)) + list(range(i + 1, D))
            omega2[idx] = omega[1:]
            l = range(i * N, (i + 1) * N)

            # random phase shift on [0, 2pi) following Saltelli et al.
            # Technometrics 1999
            phi = 2 * math.pi * np.random.rand()

            for j in range(D):
                g = 0.5 + (1 / math.pi) * \
                    np.arcsin(np.sin(omega2[j] * s + phi))
                X[l, j] = g

        self.scale_samples(X, bounds)
        return X

    def analyze(self, problem, Y, D, parnames, M=4, print_to_console=False):
        if len(Y.shape) > 1:
            Y = Y.flatten()
        print(Y.size)

        if Y.size % (D) == 0:
            N = int(Y.size / D)
        elif Y.size > D:
            N = int(Y.size / D)
            rest = Y.size - N*D
            print("""
                We can not use """ + str(rest) + """ samples which was generated
                of totaly """ + str(Y.size) + """ 
                """)
        else:
            print("""
                Error: Number of samples in model output file must be a multiple of D, 
                where D is the number of parameters in your parameter file.
              """)
            exit()

        # Recreate the vector omega used in the sampling
        omega = np.empty([D])
        omega[0] = math.floor((N - 1) / (2 * M))
        m = math.floor(omega[0] / (2 * M))

        if m >= (D - 1):
            omega[1:] = np.floor(np.linspace(1, m, D - 1))
        else:
            omega[1:] = np.arange(D - 1) % m + 1

        # Calculate and Output the First and Total Order Values
        if print_to_console:
            print("Parameter First Total")
        Si = dict((k, [None] * D) for k in ['S1', 'ST'])
        for i in range(D):
            l = np.arange(i * N, (i + 1) * N)
            Si['S1'][i] = self.compute_first_order(Y[l], N, M, omega[0])
            Si['ST'][i] = self.compute_total_order(Y[l], N, omega[0])
            if print_to_console:
                print("%s %f %f" %
                      (parnames[i], Si['S1'][i], Si['ST'][i]))
        return Si

    def compute_first_order(self, outputs, N, M, omega):
        f = np.fft.fft(outputs)
        Sp = np.power(np.absolute(f[np.arange(1, int(N / 2))]) / N, 2)
        V = 2 * np.sum(Sp)
        D1 = 2 * np.sum(Sp[np.arange(1, M + 1) * int(omega) - 1])
        return D1 / V

    def compute_total_order(self, outputs, N, omega):
        f = np.fft.fft(outputs)
        Sp = np.power(np.absolute(f[np.arange(1, int((N + 1) / 2))]) / N, 2)
        V = 2 * np.sum(Sp)
        Dt = 2 * sum(Sp[np.arange(int(omega / 2))])
        return (1 - Dt / V)

    def sample(self, repetitions):
        """
        Samples from the FAST algorithm.

        Input
        ----------
        repetitions: int 
            Maximum number of runs.  
        """
        print('Starting the FAST algotrithm with '+str(repetitions)+ ' repetitions...')
        self.set_repetiton(repetitions)
        print('Creating FAST Matrix')
        # Get the names of the parameters to analyse
        names = self.parameter()['name']
        # Get the minimum and maximum value for each parameter from the
        # distribution
        parmin, parmax = self.parameter()['minbound'], self.parameter()[
            'maxbound']

        # Create an Matrix to store the parameter sets
        N = int(math.ceil(float(repetitions) / float(len(parmin))))
        bounds = []
        for i in range(len(parmin)):
            bounds.append([parmin[i], parmax[i]])
        Matrix = self.matrix(bounds, N, M=4)
        lastbackup=0
        if self.breakpoint == 'read' or self.breakpoint == 'readandwrite':
            data_frombreak = self.read_breakdata(self.dbname)
            rep = data_frombreak[0]
            Matrix = data_frombreak[1]

        param_generator = (
            (rep, Matrix[rep]) for rep in range(len(Matrix)))
        for rep, randompar, simulations in self.repeat(param_generator):
            # Calculate the objective function
            self.postprocessing(rep, randompar, simulations)

            if self.breakpoint == 'write' or self.breakpoint == 'readandwrite':
                if rep >= lastbackup+self.backup_every_rep:
                    work = (rep, Matrix[rep:])
                    self.write_breakdata(self.dbname, work)
                    lastbackup = rep
        self.final_call()
        
        try:            
            data = self.datawriter.getdata()
            # this is likely to crash if database does not assign name 'like1'
            Si = self.analyze(
                bounds, data['like1'], len(bounds), names, print_to_console=True)
            return Si
        except AttributeError:  # Happens if no database was assigned
            pass
