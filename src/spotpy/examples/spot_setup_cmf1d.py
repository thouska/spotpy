# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Philipp Kraft

A one dimensional cmf model analysing data from the Schwingbach hydrological observatory.
You need to have cmf and pandas installed on your system: svn checkout svn://fb09-pasig.umwelt.uni-giessen.de/cmf/trunk cmf
'''

import cmf
from datetime import datetime, timedelta
import numpy as np
import spotpy
import os, sys


class _CmfProject:
    """
    This class describes the cmf setup
    """

    def __init__(self, par):
        cmf.set_parallel_threads(1)

        # run the model
        self.project = cmf.project()
        self.cell = self.project.NewCell(x=0, y=0, z=238.628, area=1000, with_surfacewater=True)
        c = self.cell
        r_curve = cmf.VanGenuchtenMualem(Ksat=10**par.pKsat, phi=par.porosity, alpha=par.alpha, n=par.n)

        # Make the layer boundaries and create soil layers
        lthickness = [.01] * 5 + [0.025] * 6 + [0.05] * 6 + [0.1] * 5
        ldepth = np.cumsum(lthickness)

        # Create soil layers and pick the new soil layers if they are inside of the evaluation depth's

        for d in ldepth:
            c.add_layer(d, r_curve)

        # Use a Richards model
        c.install_connection(cmf.Richards)

        # Use shuttleworth wallace
        self.ET = c.install_connection(cmf.ShuttleworthWallace)

        c.saturated_depth = 0.5

        self.gw = self.project.NewOutlet('groundwater', x=0, y=0, z=.9)
        cmf.Richards(c.layers[-1], self.gw)
        self.gw.potential = c.z - 0.5 #IMPORTANT
        self.gw.is_source = True
        self.gwhead = cmf.timeseries.from_scalar(c.z - 0.5)

        self.solver = cmf.CVodeIntegrator(self.project, 1e-9)

    def set_parameters(self, par):
        """
        Sets the parameters of the model
        :param par: The parameter value object (named tuple)
        :return:
        """
        try:
            for l in self.cell.layers:
                r_curve = cmf.VanGenuchtenMualem(Ksat=10**par.pKsat, phi=par.porosity, alpha=par.alpha, n=par.n)
                r_curve.w0 = r_curve.fit_w0()
                l.soil = r_curve
            self.cell.saturated_depth = 0.5
            self.gw.potential = self.cell.z - 0.5
        except RuntimeError as e:
            sys.stderr.write('Set parameters failed with:\n' + str(par) + '\n' + str(e))
            raise

    def load_meteo(self, driver_data):

        datastart = driver_data.time[0]

        # Create meteo station for project
        meteo = self.project.meteo_stations.add_station('Schwingbach', position=(0, 0, 0), tz=1, timestep=cmf.h)
        rain = cmf.timeseries.from_array(datastart, cmf.h, driver_data.rain_mmday)
        meteo.rHmean = cmf.timeseries.from_array(datastart, cmf.h, driver_data.relhum_perc)
        meteo.Windspeed = cmf.timeseries.from_array(datastart, cmf.h, driver_data.windspeed_ms)
        meteo.Rs = cmf.timeseries.from_array(datastart, cmf.h, driver_data.solarrad_wm2 * 86400e-6)
        meteo.T = cmf.timeseries.from_array(datastart, cmf.h, driver_data.airtemp_degc)
        meteo.Tmax = meteo.T.floating_max(cmf.day)
        meteo.Tmin = meteo.T.floating_min(cmf.day)

        self.project.rainfall_stations.add('Schwingbach', rain, (0, 0, 0))
        self.project.use_nearest_rainfall()


        # Use the meteorological station for each cell of the project
        self.project.use_nearest_meteo()
        self.meteo = meteo
        self.gwhead = cmf.timeseries.from_array(datastart, cmf.h, driver_data.gwhead_m)

    def set_boundary(self, t):
        """
        Sets the boundary conditions for time t
        :param t: a cmf.Time
        :return: None
        """
        gw_level = self.gwhead[t]
        if np.isfinite(gw_level):
            self.gw.potential = gw_level


def _load_data(filename):
    """
    Loads data from csv, where the first column has the measurement date and all the other columns additional data
    :return: Rec-Array with date as a first column holding datetimes
    """
    def str2date(s):
        """Converts a string to a datetime"""
        return datetime.strptime(s.decode(), '%Y-%m-%d %H:%M:%S')
    # Load the data
    return np.recfromcsv(filename, converters={0: str2date}, comments='#')


class _ProgressReporter:
    """
    Simple helper class to report progress and check for too long runtime
    """
    def __init__(self, start, end, frequency=cmf.week, max_runtime=15 * 60, verbose=False):
        self.verbose = verbose
        self.frequency = frequency
        self.max_runtime = max_runtime
        self.stopwatch = cmf.StopWatch(cmf.AsCMFtime(start), cmf.AsCMFtime(end))

    def __call__(self, t):
        elapsed, total, remaining = self.stopwatch(t)
        if self.verbose and not t % cmf.week:
            print('{modeltime:%Y-%m-%d}: {elapsed}/{total}'
                  .format(modeltime=t.AsPython(),
                          elapsed=elapsed * cmf.sec,
                          total=total * cmf.sec))
        if elapsed > self.max_runtime:
            raise RuntimeError('{:%Y-%m-%d %H:%S} model took {:0.0f}min until now, stopping'
                               .format(t.AsPython(), elapsed / 60))


class Cmf1d_Model(object):
    """
    A 1d Richards based soilmoisture model for Schwingbach site #24
    """

    alpha = spotpy.parameter.Uniform(0.0001, 0.2, optguess=0.1156, doc=u'α in 1/cm for van Genuchten Mualem model')
    pKsat = spotpy.parameter.Uniform(-2, 2, optguess=0, doc=u'log10 of saturated conductivity of the soil in m/day')
    n = spotpy.parameter.Uniform(1.08, 1.8, optguess=1.1787, doc=u'van Genuchten-Mualem n')
    porosity = spotpy.parameter.Uniform(0.3, 0.65, optguess=0.43359, doc=u'φ in m³/m³')

    def __init__(self, days=None):

        self.driver_data = _load_data('cmf_data/driver_data_site24.csv')
        self.evaluation_data = np.loadtxt('cmf_data/soilmoisture_site24.csv', delimiter=',',
                                          comments='#', usecols=[1,2,3])

        self.datastart = self.driver_data.time[0]

        # Set the end point for the model runtime, either by parameter days or
        # run to the end of available driver data
        if days is None:
            self.dataend = self.driver_data.time[-1]
        else:
            self.dataend = self.datastart + timedelta(days=days)

        # The depth below ground in m where the evaluation data belongs to
        self.eval_depth = [0.1, 0.25, 0.4]


        # Make the model
        self.model = _CmfProject(self.make_parameters())
        # Load meteo data
        self.model.load_meteo(driver_data=self.driver_data)
        self.__doc__ += '\n\n' + cmf.describe(self.model.project)

    def make_parameters(self, random=False, **kwargs):
        return spotpy.parameter.create_set(self)

    def evaluation(self):
        """
        :return: The evaluation soilmoisture as a 2d array 
        """

        return self.evaluation_data

    def objectivefunction(self, simulation, evaluation):
        """
        Returns the negative RMSE for all values
        :param simulation: 
        :param evaluation: 
        :return: 
        """

        # Find all data positions where simulation and evaluation data is present
        take = np.isfinite(simulation) & np.isfinite(evaluation)
        rmse = -spotpy.objectivefunctions.rmse(evaluation=evaluation[take], simulation=simulation[take])
        return rmse

    def get_eval_layers(self, eval_depth=None):
        """
        Returns the list of layers, that should be evaluated

        :param eval_depth: List of depth below ground where evaluation should be done. If None, all layers are evaluated
        :return:
        """
        c = self.model.cell
        if eval_depth:
            edi = 0 # current index of the evaluation depth
            eval_layers = [] # the layers to do the evaluation are stored here
            for l in c.layers:
                if edi < len(self.eval_depth) and l.upper_boundary <= self.eval_depth[edi] < l.lower_boundary:
                    eval_layers.append(l)
                    edi += 1
        else:
            eval_layers = list(c.layers)
        return eval_layers

    def simulation(self, par=None, verbose=False):
        '''
        Runs the model instance
        
        :param par: A namedtuple of model parameters
        :param verbose: Write the progress for the single run
        :return: A 2d array of simulated soil moisture values per depth
        '''

        # Set the parameters
        par = par or self.make_parameters()
        self.model.set_parameters(par)
        if verbose:
            print('Parameters:')
            for k, v in par._asdict().items():
                print('    {} = {:0.4g}'.format(k,v))

        eval_layers = self.get_eval_layers(self.eval_depth)
        # Prepare result array with nan's
        result = np.nan * np.ones(shape=(len(self.evaluation_data), len(eval_layers)))
        # Save the starting conditions
        result[0] = [l.theta for l in eval_layers]
        reporter = _ProgressReporter(self.datastart, self.dataend, max_runtime=15 * 60, verbose=verbose)

        try:
            for i, t in enumerate(self.model.solver.run(self.datastart, self.dataend, timedelta(hours=1))):
                self.model.set_boundary(t)
                # Get result
                result[i+1] = [l.theta for l in eval_layers]
                # Raises if the runtime is too long
                reporter(t)
        except RuntimeError as e:
            sys.stderr.write(str(par))
            sys.stderr.write(str(e))

        return result


if __name__ == '__main__':
    print(spotpy.__file__, spotpy.__version__)
    if len(sys.argv) > 1:
        runs = int(sys.argv[1])
    else:
        runs = 1

    model = Cmf1d_Model()
    print(spotpy.describe.setup(model))
    if runs > 1:
        parallel = 'mpi' if 'OMPI_COMM_WORLD_SIZE' in os.environ else 'seq'
        sampler = spotpy.algorithms.mc(model,
                                        dbformat='csv',
                                        dbname='cmf1d_mc',
                                        parallel=parallel,
                                        save_sim=False)
        sampler.sample(runs)
    else:
        from spotpy.gui.mpl import GUI
        gui = GUI(model)
        gui.show()
