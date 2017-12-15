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

        # Get be, step and end from the date column
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
    Enthält das gesamte Modell
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
        Initialisiert das Modell und baut das Grundsetup zusammen

        MP111 - Änderungsbedarf: Sehr groß, hier wird es Euer Modell definiert
        Verständlichkeit: mittel

        """

        # TODO: Parameterliste erweitern und anpassen.
        # Wichtig: Die Namen müssen genau die gleichen sein,
        # wie die Argumente in der Funktion setparameters.
        #
        # Parameter werden wie folgt definiert:
        # param(<Name>,<Minimum>,<Maximum>)

        self.dbname = 'philipp-singlestorage'

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

        # Verdunstung
        cmf.HargreaveET(layer, c.transpiration)

        # Outlet
        outlet = p.NewOutlet('outlet', 10, 0, 0)

        return p, outlet

    def simulation(self, vector=None, verbose=False):
        """
        SpotPy erwartet eine Methode simulation. Diese methode ruft einfach
        setparameters und runmodel auf, so dass spotpy zufrieden ist
        """
        self.setparameters(vector)
        result_q = self.runmodel(verbose)

        return np.array(result_q)

    def setparameters(self, par=None):
        """
        Setzt die Parameter, dabei werden parametrisierte Verbindungen neu erstellt

        MP111 - Änderungsbedarf: Sehr groß, hier werden alle Parameter des Modells gesetzt
        Verständlichkeit: mittel

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

        # TODO: Alle weiteren Connections / Parameter von Eurem Modell aufbauen

    def runmodel(self, verbose=False):
        """
        Startet das Modell

        verbose = Wenn verbose = True, dann wird zu jedem Tag etwas ausgegeben

        MP111 - Änderungsbedarf: Gering, kann so bleiben, kann aber auch
                        um andere Ergebniszeitreihen ergänzt werden. Achtung,
                        falls ihr mehrere Outlets benutzt
        Verständlichkeit: gut

        Reference: http://fb09-pasig.umwelt.uni-giessen.de/cmf/wiki/CmfTutFluxes
        """
        # Erzeugt ein neues Lösungsverfahren
        solver = cmf.ImplicitEuler(self.project, 1e-9)
        # Verkürzte schreibweise für die Zelle - spart tippen
        c = self.project[0]

        # Neue Zeitreihe für Modell-Ergebnisse (res - result)
        res_q = cmf.timeseries(self.begin, cmf.day)
        # Starte den solver und rechne in Tagesschritten
        for t in solver.run(self.data.begin, self.end, cmf.day):
            # Fülle die Ergebnisse
            if t >= self.begin:
                res_q.add(self.outlet.waterbalance(t))
            # Gebe eine Statusmeldung auf den Bildschirm aus,
            # dann weiß man wo der solver gerade ist
            if verbose:
                print(t, 'P={:5.3f}'.format(c.get_rainfall(t)))
        # Gebe die gefüllten Ergebnis-Zeitreihen zurück
        return res_q

    def objectivefunction(self, simulation, evaluation):
        """
        Gehört auch zum spotpy-Interface. 

        MP111 - Änderungsbedarf: Keiner
        Verständlichkeit: Mittel
        """
        return spotpy.objectivefunctions.nashsutcliffe(evaluation, simulation)

    def evaluation(self):
        """
        Gehört auch zum spotpy-Interface. 

        MP111 - Änderungsbedarf: Keiner
        Verständlichkeit: Schlecht
        """
        runoff_mm = self.data.runoff_mm(self.area)
        return np.array(runoff_mm[self.begin:self.end + datetime.timedelta(days=1)])


# http://stackoverflow.com/questions/419163/what-does-if-name-main-do
if __name__ == '__main__':

    # Importiere Algorithmus
    from spotpy.algorithms import mc as Sampler

    # Finde heraus, ob das ganze parallel laufen soll (für Supercomputer)
    parallel = 'mpi' if 'OMPI_COMM_WORLD_SIZE' in os.environ else 'seq'

    # Create the spotted model
    model = SingleStorage(datetime.datetime(1980, 1, 1),
                          datetime.datetime(1985, 12, 31))

    if 'MP111RUNS' in os.environ:
        runs = int(os.environ['MP111RUNS'])

    elif len(sys.argv) > 1:
        runs = int(sys.argv[1])
    else:
        runs = 1

    sampler = Sampler(model, parallel=parallel, dbname=model.dbname, dbformat='csv', save_sim=True)

    print(spotpy.describe.describe(sampler))

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


