# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Jingshui Huang, Tobias Houska and the SALib team
"""

import math

import numpy as np

from . import _algorithm


class morris(_algorithm):
    """
    Morris Screening Sensitivity Test (FAST)

    This class holds the Morris Screening Sensitivity Test (MORRIS) based on Morris (1991), Campolongo et al (2007) and Ruano et al. (2012):

    Morris, M.D., 1991, Factorial Sampling Plans for Preliminary Computational Experiments. Technometrics 33, 161-174.

    Campolongo, F., Cariboni, J., & Saltelli, A. 2007. An effective screening design for sensitivity analysis of large
    models. Environmental Modelling & Software, 22(10), 1509-1518.

    Ruano, M.V., Ribes, J., Seco, A., Ferrer, J., 2012. An improved sampling strategy based on trajectory design for
    application of the Morris method to systems with many input factors. Environmental Modelling & Software 37, 103-109.

    The presented code is based on SALib
    Copyright (C) 2013-2015 Jon Herman and others. Licensed under the GNU Lesser General Public License.
    The Sensitivity Analysis Library is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
    The Sensitivity Analysis Library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License along with the Sensitivity Analysis Library. If not, see http://www.gnu.org/licenses/.
    """

    def __init__(self, *args, **kwargs):
        """
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
            *False: Simulation results will not be saved
        """
        kwargs["algorithm_name"] = "Morris Screening Sensitivity Test (MORRIS)"
        super(morris, self).__init__(*args, **kwargs)

    def scale_samples(self, params, bounds):
        """
        Rescales samples in 0-to-1 range to arbitrary bounds.

        Arguments:
            bounds - list of lists of dimensions num_params-by-2
            params - numpy array of dimensions num_params-by-N,
            where N is the number of samples
        """
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
        np.add(
            np.multiply(params, (upper_bounds - lower_bounds), out=params),
            lower_bounds,
            out=params,
        )

    def matrix(self, bounds, problem, N, M=4):
        from SALib.sample import morris as morris_sample

        X = morris_sample.sample(
            problem,
            N=N,
            num_levels=M,
            optimal_trajectories=None,  # set int to use Campolongo opt design
            local_optimization=True,  # Ruano-style distance improvement
            seed=123,
        )

        self.scale_samples(X, bounds)
        return X

    def analyze(self, problem, X, Y, num_levels=4, print_to_console=False):
        from SALib.analyze import morris as morris_analyze

        Si = morris_analyze.analyze(
            problem,
            X,
            Y,
            conf_level=0.95,
            num_levels=num_levels,
            print_to_console=print_to_console,
        )
        return Si

    def sample(self, repetitions, num_levels=4):
        """
        Samples from the MORRIS algorithm.

        Input
        ----------
        repetitions: int
            Maximum number of runs.
        """
        self.set_repetiton(repetitions)
        print(
            "Starting the MORRIS algorithm with " + str(repetitions) + " repetitions..."
        )
        print("Creating MORRIS Matrix")
        # Get the names of the parameters to analyse
        names = self.parameter()["name"]
        # Get the minimum and maximum value for each parameter from the
        # distribution
        parmin, parmax = self.parameter()["minbound"], self.parameter()["maxbound"]

        bounds = []
        for i in range(len(parmin)):
            bounds.append([parmin[i], parmax[i]])
        # -----------------------------------------------------------------------------
        # MORRIS DESIGN SETTINGS
        # -----------------------------------------------------------------------------
        # Assume df_params from your existing script
        # columns: name, change_type, lower_bound, upper_bound, subbasins
        problem = {
            "num_vars": len(names),
            "names": names.tolist(),  # or name_spotpy if you expanded names
            "bounds": bounds,
            #'bounds': self.parameter()[['minbound','maxbound']].values.tolist()
        }

        k = problem["num_vars"]
        N = 15  # number of trajectories (adjust for robustness vs runtime)
        num_levels = num_levels  # classic Morris grid

        # Create an Matrix to store the parameter sets
        N = int(math.ceil(float(repetitions) / float(len(parmin))))
        bounds = []
        for i in range(len(parmin)):
            bounds.append([parmin[i], parmax[i]])
        Matrix = self.matrix(bounds, problem, N, M=num_levels)
        lastbackup = 0
        if self.breakpoint == "read" or self.breakpoint == "readandwrite":
            data_frombreak = self.read_breakdata(self.dbname)
            rep = data_frombreak[0]
            Matrix = data_frombreak[1]

        param_generator = ((rep, Matrix[rep]) for rep in range(len(Matrix)))
        for rep, randompar, simulations in self.repeat(param_generator):
            # Calculate the objective function
            self.postprocessing(rep, randompar, simulations)

            if self.breakpoint == "write" or self.breakpoint == "readandwrite":
                if rep >= lastbackup + self.backup_every_rep:
                    work = (rep, Matrix[rep:])
                    self.write_breakdata(self.dbname, work)
                    lastbackup = rep
        self.final_call()

        try:
            data = self.datawriter.getdata()
            # this is likely to crash if database does not assign name 'like1'
            Si = self.analyze(
                problem,
                Matrix,
                data["like1"],
                num_levels=num_levels,
                print_to_console=False,
            )
            # Si = self.analyze(
            #     bounds, data["like1"], len(bounds), names, M=num_levels, print_to_console=True
            # )
            print(Si["mu_star"])
        except AttributeError:  # Happens if no database was assigned
            pass
