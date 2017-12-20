#!/usr/bin/env python
# coding: utf-8

"""

"""

from __future__ import division, print_function
import sys
import datetime
# noinspection PyPackageRequirements
import cmf
import spotpy
from spotpy.parameter import Uniform
import os

# noinspection PyPackageRequirements
import numpy as np
np.seterr(all='ignore')


class DataProvider:
    """
    Holds the forcing and calibration data
    """

    def __init__(self):

        # Load data from file using numpy magic
        data = np.recfromcsv('cmf_data/fulda_climate.csv')

        def bstr2date(bs):
            """Helper function to convert date byte string to datetime object"""
            return datetime.datetime.strptime(bs.decode(), '%d.%m.%Y')

        # Get begin, step and end from the date column
        self.begin = bstr2date(data.date[0])
        self.step = bstr2date(data.date[1]) - self.begin
        self.end = bstr2date(data.date[-1])

        def a2ts(a):
            """Converts an array column to a timeseries"""
            return cmf.timeseries.from_array(self.begin, self.step, a)

        self.P = a2ts(data.prec)
        self.T = a2ts(data.tmean)
        self.Tmin = a2ts(data.tmin)
        self.Tmax = a2ts(data.tmax)
        self.Q = a2ts(data.q)


    def runoff_mm(self, area):
        sec_per_day = 86400
        mm_per_m = 1000
        return self.Q * sec_per_day / area * mm_per_m

    def add_stations(self, project):
        """
        Creates a rainstation and a meteo station for the cmf project
        :param project: A cmf.project
        :return: rainstation, meteo
        """
        rainstation = project.rainfall_stations.add('Grebenau avg', self.P, (0, 0, 0))

        project.use_nearest_rainfall()

        # Temperaturdaten
        meteo = project.meteo_stations.add_station('Grebenau avg', (0, 0, 0))
        meteo.T = self.T
        meteo.Tmin = self.Tmin
        meteo.Tmax = self.Tmax

        project.use_nearest_meteo()

        return rainstation, meteo


# noinspection PyMethodMayBeStatic
class SingleStorage:
    """
    A simple hydrological single storage model.
    No snow, interception  or routing.
    """
    # Catchment area
    area = 2976.41e6  # sq m
    # General storage parameter
    V0 = Uniform(10, 10000, 1000)

    # ET parameters
    fETV1 = Uniform(0.01, 1, 0.2, doc='if V<fETV1*V0, water uptake stress for plants starts')
    fETV0 = Uniform(0, 0.9, 0.2, doc='if V<fETV0*fETV1*V0, plants die of drought')

    # Outflow parameters
    tr = Uniform(0.1, 1000, doc='Residence time of water in storage when V=V0')
    Vr = Uniform(0, 1, 0.0, doc='Residual water in storage in terms of V0')
    beta = Uniform(0.3, 5, 1, doc='Exponent in kinematic wave function')

    def __init__(self, begin=None, end=None):
        """
        Initializes the model

        :param begin: Start year for calibration
        :param end: stop year

        """

        self.dbname = 'cmf-singlestorage'

        # Loads driver data
        self.data = DataProvider()
        self.project, self.outlet = self.create_project()
        self.data.add_stations(self.project)
        self.setparameters()

        self.begin = begin or self.data.begin
        self.end = end or self.data.end

    def create_project(self):
        """
        Creates the cmf project with its basic elements
        """
        # Use only a single thread, that is better for a calibration run and for small models
        cmf.set_parallel_threads(1)

        # make the project
        p = cmf.project()

        # make a new cell
        c = p.NewCell(0, 0, 0, 1000)

        # Add a storage
        layer = c.add_layer(1.0)
        # TODO: add more layers
        # ET
        cmf.HargreaveET(layer, c.transpiration)
        # Outlet
        outlet = p.NewOutlet('outlet', 10, 0, 0)
        return p, outlet

    def setparameters(self, par=None):
        """
        Sets the parameters of the model by creating the connections
        """
        par = par or spotpy.parameter.create_set(self)

        # Some shortcuts to gain visibility
        c = self.project[0]
        o = self.outlet

        # Set uptake stress
        ETV1 = par.fETV1 * par.V0
        ETV0 = par.fETV0 * ETV1
        c.set_uptakestress(cmf.VolumeStress(ETV1, ETV0))

        # Connect layer with outlet
        cmf.PowerLawConnection(c.layers[0], o,
                               Q0=par.V0 / par.tr, beta=par.beta,
                               residual=par.Vr * par.V0, V0=par.V0)


    def runmodel(self, verbose=False):
        """
        Runs the model and saves the results
        """
        solver = cmf.ImplicitEuler(self.project, 1e-9)
        c = self.project[0]

        # result timeseries
        res_q = cmf.timeseries(self.begin, cmf.day)

        # start solver and calculate in daily steps
        for t in solver.run(self.data.begin, self.end, cmf.day):
            # append results
            res_q.add(self.outlet.waterbalance(t))
            # Give the status the screen to let us know what is going on
            if verbose:
                print(t, 'P={:5.3f}'.format(c.get_rainfall(t)))

        return res_q

    def simulation(self, vector=None, verbose=False):
        """
        Sets the parameters of the model and starts a run
        :return: np.array with runoff in mm/day
        """
        self.setparameters(vector)
        result_q = self.runmodel(verbose)

        return np.array(result_q[self.begin:self.end])

    def objectivefunction(self, simulation, evaluation):
        """
        Calculates the goodness of the simulation
        """
        return spotpy.objectivefunctions.nashsutcliffe(evaluation, simulation)

    def evaluation(self):
        """
        Returns the evaluation data
        """
        runoff_mm = self.data.runoff_mm(self.area)
        return np.array(runoff_mm[self.begin:self.end])


# http://stackoverflow.com/questions/419163/what-does-if-name-main-do
if __name__ == '__main__':

    # Get the Monte-Carlo sampler
    from spotpy.algorithms import mc as Sampler

    # Check if we are running on a supercomputer or local
    parallel = 'mpi' if 'OMPI_COMM_WORLD_SIZE' in os.environ else 'seq'

    # Create the model
    model = SingleStorage(datetime.datetime(1980, 1, 1),
                          datetime.datetime(1985, 12, 31))

    # Get number of runs
    if 'SPOTPYRUNS' in os.environ:
        # from environment
        runs = int(os.environ['SPOTPYRUNS'])
    elif len(sys.argv) > 1:
        # from command line
        runs = int(sys.argv[1])
    else:
        # run once
        runs = 1

    # Create the sampler
    sampler = Sampler(model, parallel=parallel, dbname=model.dbname, dbformat='csv', save_sim=True)

    # Print our configuration
    print(spotpy.describe.describe(sampler))
    # Print the cmf setup
    print(cmf.describe(model.project))

    # Do the sampling
    if runs > 1:
        # Now we can sample with the implemented Monte Carlo algortihm:
        sampler.sample(runs)
    else:
        result = model.simulation(verbose=True)
        for name, value in spotpy.objectivefunctions.calculate_all_functions(model.evaluation(), result):
            try:
                print('{:>30.30s} = {:0.6g}'.format(name, value))
            except ValueError:
                print('{:>30.30s} = {}'.format(name, value))


