'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Philipp Kraft

A one dimensional cmf model analysing data from the Schwingbach hydrological observatory.
You need to have cmf and pandas installed on your system: svn checkout svn://fb09-pasig.umwelt.uni-giessen.de/cmf/trunk cmf
'''

import pandas as pd
import cmf
from datetime import timedelta
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
        # lthickness = [0.01] * 100
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

        self.solver = cmf.CVodeIntegrator(self.project, 1e-9)

    def set_parameters(self, par):
        try:
            for l in self.cell.layers:
                r_curve = cmf.VanGenuchtenMualem(Ksat=10**par.pKsat, phi=par.porosity, alpha=par.alpha, n=par.n)
                r_curve.w0 = r_curve.fit_w0()
                l.wetness = 0.9
                l.soil = r_curve
            self.cell.saturated_depth = 0.5
            self.gw.potential = self.cell.z - 0.5
        except RuntimeError as e:
            sys.stderr.write('Set parameters failed with:\n' + str(par) + '\n' + str(e))
            raise

    def load_meteo(self, driver_data):
        datastart = driver_data.index[0].to_datetime()

        # Create meteo station for project
        meteo = self.project.meteo_stations.add_station('Schwingbach', position=(0, 0, 0), tz=1, timestep=cmf.h)
        rain = cmf.timeseries.from_array(datastart, cmf.h, driver_data.rain_mmday)
        meteo.rHmean = cmf.timeseries.from_array(datastart, cmf.h, driver_data.relhum_perc)
        meteo.Windspeed = cmf.timeseries.from_array(datastart, cmf.h, driver_data.windspeed_ms)
        meteo.Rs = cmf.timeseries.from_array(datastart, cmf.h, driver_data.solarrad_Wm2 * 86400e-6)
        meteo.T = cmf.timeseries.from_array(datastart, cmf.h, driver_data.airtemp_degC)
        meteo.Tmax = meteo.T.floating_max(cmf.day)
        meteo.Tmin = meteo.T.floating_min(cmf.day)

        self.project.rainfall_stations.add('Schwingbach', rain, (0, 0, 0))
        self.project.use_nearest_rainfall()
        # Use the meteorological station for each cell of the project
        self.project.use_nearest_meteo()


class Cmf1d_Model(object):
    """
    A 1d Richards based soilmoisture model for Schwingbach site #24
    """

    alpha = spotpy.parameter.Uniform(0.0001, 0.2, optguess=0.1156, doc='α in 1/cm for van Genuchten Mualem model')
    pKsat = spotpy.parameter.Uniform(-2, 2, optguess=0, doc='log10 of saturated conductivity of the soil in m/day')
    n = spotpy.parameter.Uniform(1.08, 1.8, optguess=1.1787, doc='van Genuchten-Mualem n')
    porosity = spotpy.parameter.Uniform(0.3, 0.65, optguess=0.43359, doc='Porosity in m³/m³')

    def __init__(self, days=None):

        self.driver_data = pd.read_csv('data/driver_data_site24.csv',
                                        parse_dates=[0], index_col=[0])
        self.evaluation_data = pd.read_csv('data/soilmoisture_site24.csv',
                                        parse_dates=[0], index_col=[0])
        self.datastart = self.driver_data.index[0].to_datetime()
        if days is None:
            self.dataend = self.driver_data.index[-1].to_datetime()
        else:
            self.dataend = self.datastart + timedelta(days=days)


        self.eval_depth = [0.1,0.25,0.4]


        # Make the model
        self.model = _CmfProject(self.optguess())
        self.model.load_meteo(driver_data=self.driver_data)

    def __str__(self):
        mname = type(self).__name__
        doc = self.__doc__
        params = '\n'.join(' - {p}'.format(p=p) for p in spotpy.parameter.get_parameters_from_class(type(self)))
        return '{mname}\n{doc}\n\nParameters:\n{params}'.format(mname=mname,doc=doc,params=params)


    def optguess(self):
        """
        :return: a parameter value object for the default values (optguess) of the parameters 
        """
        params = spotpy.parameter.get_parameters_from_class(type(self))
        partype = spotpy.parameter.get_namedtuple_from_paramnames(type(self).__name__,
                                                                  [p.name.encode() for p in params])
        return partype(*spotpy.parameter.generate(params)['optguess'])

    def get_param_value(self, **kwargs):
        params = spotpy.parameter.get_parameters_from_class(type(self))
        partype = spotpy.parameter.get_namedtuple_from_paramnames(type(self).__name__,
                                                                  [p.name.encode() for p in params])
        pardict = self.optguess()._asdict()
        pardict.update(kwargs)
        return partype(**pardict)

    def evaluation(self):
        """
        
        :return: The evaluation soilmoisture as a 2d array 
        """
        return np.array(self.evaluation_data, dtype=float)

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

    def simulation(self, par, verbose=False):
        '''
        Runs the model instance
        
        :param par: A namedtuple of model parameters
        :param verbose: Write the progress for the single run
        :return: A 2d array of simulated soil moisture values per depth
        '''

        # Set the parameters
        self.model.set_parameters(par)
        if verbose:
            print('Parameters:')
            for k, v in par._asdict().items():
                print('    {} = {:0.4g}'.format(k,v))

        c = self.model.cell

        # Get evaluation layers
        if self.eval_depth:
            edi = 0 # current index of the evaluation depth
            eval_layers = [] # the layers to do the evaluation are stored here
            for l in c.layers:
                if edi < len(self.eval_depth) and l.upper_boundary <= self.eval_depth[edi] < l.lower_boundary:
                    eval_layers.append(l)
                    edi += 1
        else:
            eval_layers = list(c.layers)

        result = np.ones(shape=(len(self.evaluation_data), len(eval_layers))) * np.nan
        # Save first
        result[0] = [l.theta for l in eval_layers]
        stopwatch = cmf.StopWatch(cmf.AsCMFtime(self.datastart), cmf.AsCMFtime(self.dataend))
        try:
            for i, t in enumerate(self.model.solver.run(self.datastart, self.dataend, timedelta(hours=1))):
                pt = t.AsPython()
                gw_level = self.driver_data.gwhead_m.loc[pt]
                # Set boundary condition
                if np.isfinite(gw_level):
                    self.model.gw.potential = gw_level
                # Get result
                result[i+1] = [l.theta for l in eval_layers]

                elapsed, total, remaining = stopwatch(t)
                if verbose and not t % cmf.week:
                    print('{pt:%Y-%m-%d}: {elapsed}/{total}, theta={theta:0.4f}'
                          .format(pt=pt, elapsed=elapsed*cmf.sec, total=total*cmf.sec, theta=c.layers.theta.mean()))
                if elapsed > 60 * 15:
                    raise RuntimeError('{:%Y-%m-%d %H:%S} model took more than {:0.0f}min until now, stopping'
                                       .format(pt, elapsed / 60))
        except KeyboardInterrupt:
            pass
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
    print(model)
    if runs > 1:
        # Finde heraus, ob das ganze parallel laufen soll (für Supercomputer)
        parallel = 'mpi' if 'OMPI_COMM_WORLD_SIZE' in os.environ else 'seq'
        sampler = spotpy.algorithms.mc(model,
                                        dbformat='csv',
                                        dbname='cmf1d_mc',
                                        parallel=parallel,
                                        save_sim=False)
        sampler.sample(runs)
    else:
        # model.eval_depth = None
        par = model.optguess()
        result = model.simulation(par, verbose=True)
        rmse = model.objectivefunction(result, model.evaluation())
        print('Model ready, RMSE={:0.4f}% soil moisture'.format(rmse))
        np.save('result.npy', result)
        np.save('eval.npy', model.evaluation())