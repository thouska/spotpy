"""
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This example implements the Rosenbrock function into SPOT.
"""
import numpy as np

import spotpy
from spotpy.objectivefunctions import rmse


class spot_setup(object):
    a = spotpy.parameter.Uniform(low=0, high=1)
    b = spotpy.parameter.Uniform(low=0, high=1)

    def __init__(self):

        self.db_headers = ["obj_functions", "parameters", "simulations"]

        self.database = open("MyOwnDatabase.txt", "w")
        self.database.write("\t".join(self.db_headers) + "\n")

    def simulation(self, vector):
        x = np.array(vector)
        simulations = [
            sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
        ]
        return simulations

    def evaluation(self):
        observations = [0]
        return observations

    def objectivefunction(self, simulation, evaluation):
        objectivefunction = -rmse(evaluation=evaluation, simulation=simulation)
        return objectivefunction

    def save(self, objectivefunctions, parameter, simulations, *args, **kwargs):
        param_str = "\t".join((str(p) for p in parameter))
        sim_str = "\t".join((str(s) for s in simulations))
        line = "\t".join([str(objectivefunctions), param_str, sim_str]) + "\n"
        self.database.write(line)


if __name__ == "__main__":
    spot_setup = spot_setup()

    # set dbformat to custom and spotpy will return results in spot_setup.save function
    sampler = spotpy.algorithms.mc(spot_setup, dbformat="custom")
    sampler.sample(
        100
    )  # Choose equal or less repetitions as you have parameters in your List
    spot_setup.database.close()  # Close the created txt file
