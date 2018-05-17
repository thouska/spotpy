# -*- coding: utf-8 -*-
'''
Copyright (c) 2018 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska, Benjamin Manns


'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy

import unittest
import numpy as np
import spotpy.analyser
import os
import pickle


class TestAnalyser(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.rep = 100
        if not os.path.isfile("setUp_pickle_file"):
            from spotpy.examples.spot_setup_rosenbrock import spot_setup
            spot_setup_object = spot_setup()
            self.results = []
            parallel = "seq"
            dbformat = "ram"
            timeout = 5

            self.sampler = spotpy.algorithms.dream(spot_setup_object, parallel=parallel, dbname='RosenDREAM', dbformat=dbformat,
                                              sim_timeout=timeout)

            self.sampler.sample(self.rep)
            self.results.append(self.sampler.getdata())
            self.results = np.array(self.results)

    def test_get_parameters(self):

        self.assertEqual(len(spotpy.analyser.get_parameters(
            self.results
        )[0][0]), 3)

    def test_get_parameternames(self):
        self.assertEqual(spotpy.analyser.get_parameternames(
            self.results
        ),['x', 'y', 'z'])

    def test_get_parameter_fields(self):
        self.assertEqual(len(spotpy.analyser.get_parameternames(
            self.results
        )), 3)

    def test_get_minlikeindex(self):
        minlikeindex = spotpy.analyser.get_minlikeindex(
            self.results
        )
        self.assertEqual(len(minlikeindex), 2)
        self.assertEqual(type(minlikeindex),type((1,1)))

    def test_get_maxlikeindex(self):
        get_maxlikeindex = spotpy.analyser.get_maxlikeindex(
            self.results
        )
        self.assertEqual(len(get_maxlikeindex), 2)
        self.assertEqual(type(get_maxlikeindex),type((1,1)))

    def test_get_like_fields(self):
        get_like_fields = spotpy.analyser.get_like_fields(
            self.results
        )
        self.assertEqual(len(get_like_fields), 1)
        self.assertEqual(type(get_like_fields),type([]))

    def test_calc_like(self):
        calc_like = spotpy.analyser.calc_like(
            self.results,
            self.sampler.evaluation,spotpy.objectivefunctions.rmse)
        self.assertEqual(len(calc_like), 1)
        self.assertEqual(type(calc_like), type([]))

    def test_get_best_parameterset(self):
        get_best_parameterset_true = spotpy.analyser.get_best_parameterset(
            self.results,True)
        get_best_parameterset_false = spotpy.analyser.get_best_parameterset(
            self.results, False)
        self.assertEqual(len(get_best_parameterset_true[0]), 3)
        self.assertEqual(type(get_best_parameterset_true[0]), np.void)
        self.assertEqual(len(get_best_parameterset_false[0]), 3)
        self.assertEqual(type(get_best_parameterset_false[0]), np.void)

    def test_get_modelruns(self):
        get_modelruns = spotpy.analyser.get_modelruns(
            self.results
        )
        self.assertEqual(len(get_modelruns[0][0]), 1)
        self.assertEqual(type(get_modelruns[0][0]), np.void)

    def test_get_header(self):
        get_header = spotpy.analyser.get_header(
            self.results
        )
        self.assertEqual(len(get_header), 6)
        self.assertEqual(type(get_header), type(()))

    def test_get_min_max(self):
        from spotpy.examples.spot_setup_ackley import spot_setup as sp_ackley
        sp_ackley = sp_ackley()
        get_min_max = spotpy.analyser.get_min_max(spotpy_setup=sp_ackley)
        self.assertEqual(len(get_min_max[0]), 30)
        self.assertEqual(type(get_min_max), type(()))

    def test_get_parbounds(self):
        from spotpy.examples.spot_setup_ackley import spot_setup as sp_ackley
        sp_ackley = sp_ackley()
        get_parbounds = spotpy.analyser.get_parbounds(spotpy_setup=sp_ackley)

        self.assertEqual(len(get_parbounds[0]), 2)
        self.assertEqual(len(get_parbounds), 30)
        self.assertEqual(type(get_parbounds), type([]))

    def test_get_percentiles(self):
        get_percentiles = spotpy.analyser.get_percentiles(
            self.results
        )
        self.assertEqual(len(get_percentiles),5)
        self.assertEqual(type(get_percentiles[0][0]), type(np.abs(-1.0)))
        self.assertEqual(type(get_percentiles),type(()))

    def test__geweke(self):
        sample1 = []
        for a in self.results:
            sample1.append(a[0])

        _geweke = spotpy.analyser._Geweke(sample1)

        self.assertEqual(len(_geweke), 20)
        self.assertEqual(type(_geweke), type(np.array([])))

    def test_plot_Geweke(self):
        sample1 = []
        for a in self.results[0]:
            sample1.append(a[0])
        spotpy.analyser.plot_Geweke(sample1,"sample1")
        fig_name = "test_plot_Geweke.png"
        plt.savefig(fig_name)

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_get_sensitivity_of_fast(self):
        from spotpy.examples.spot_setup_rosenbrock import spot_setup
        spot_setup_object = spot_setup()
        results = []
        parallel = "seq"
        dbformat = "ram"
        timeout = 5
        self.sampler = spotpy.algorithms.fast(spot_setup_object, parallel=parallel,
                                               dbname='test_get_sensitivity_of_fast', dbformat=dbformat,
                                               sim_timeout=timeout)
        self.sampler.sample(200)
        results.append(self.sampler.getdata())
        results = np.array(results)[0]
        get_sensitivity_of_fast = spotpy.analyser.get_sensitivity_of_fast(results)
        self.assertEqual(len(get_sensitivity_of_fast), 2)
        self.assertEqual(len(get_sensitivity_of_fast["S1"]), 3)
        self.assertEqual(len(get_sensitivity_of_fast["ST"]), 3)
        self.assertEqual(type(get_sensitivity_of_fast), type({}))

    def test_get_simulation_fields(self):
        get_simulation_fields = spotpy.analyser.get_simulation_fields(
            self.results
        )
        self.assertEqual(['simulation_0'],get_simulation_fields)

    def test_compare_different_objectivefunctions(self):

        from spotpy.examples.spot_setup_hymod_python import spot_setup as sp
        sp = sp()

        sampler = spotpy.algorithms.dream(sp, parallel="seq", dbname='test_compare_different_objectivefunctions',
                                          dbformat="ram",
                                          sim_timeout=5)

        sampler_mcmc = spotpy.algorithms.mcmc(sp, parallel="seq", dbname='test_compare_different_objectivefunctions',
                                          dbformat="ram",
                                          sim_timeout=5)

        sampler.sample(50)
        sampler_mcmc.sample(50)
        compare_different_objectivefunctions = spotpy.analyser.compare_different_objectivefunctions(
            sampler_mcmc.bestlike, sampler.bestlike)


        self.assertEqual(type(compare_different_objectivefunctions[1]),type(np.array([0.5])[0]))

    def test_plot_parameter_uncertainty(self):
        posterior = spotpy.analyser.get_posterior(self.results,percentage=10)
        self.assertAlmostEqual(len(posterior)/100, self.rep/1000, 1)
        self.assertEqual(type(posterior), type(np.array([])))
        spotpy.analyser.plot_parameter_uncertainty(posterior,
                                                   self.sampler.evaluation)

        fig_name = "test_plot_parameter_uncertainty.png"
        plt.savefig(fig_name)

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        # tidy up all
        os.remove(fig_name)

    def test_plot_fast_sensitivity(self):

        from spotpy.examples.spot_setup_rosenbrock import spot_setup
        spot_setup_object = spot_setup()
        parallel = "seq"
        dbformat = "ram"
        timeout = 5

        sampler = spotpy.algorithms.fast(spot_setup_object, parallel=parallel,
                                          dbname='test_get_sensitivity_of_fast', dbformat=dbformat,
                                          sim_timeout=timeout)
        sampler.sample(300)
        results = []
        results.append(sampler.getdata())
        results = np.array(results)[0]
        print("Sampler is done with")
        print(results)
        spotpy.analyser.plot_fast_sensitivity(results)

        fig_name = "FAST_sensitivity.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        # tidy up all
        os.remove(fig_name)


    def setup_griewank(self):
        if not os.path.isfile("setup_griewank_pickle"):
            from spotpy.examples.spot_setup_griewank import spot_setup
            # Create samplers for every algorithm:
            spot_setup = spot_setup()
            rep = 100
            timeout = 10  # Given in Seconds

            parallel = "seq"
            dbformat = "csv"

            sampler = spotpy.algorithms.mc(spot_setup, parallel=parallel, dbname='RosenMC', dbformat=dbformat,
                                           sim_timeout=timeout)
            sampler.sample(rep)

            fl = open("setup_griewank_pickle", "wb")
            pickle.dump({"getdata": sampler.getdata(), "evaluation": sampler.evaluation}, fl)
            fl.close()

        with open("setup_griewank_pickle", "rb") as file:
            return pickle.load(file)

    def test_plot_heatmap_griewank(self):
        fig_name = "test.png"
        spotpy.analyser.plot_heatmap_griewank([self.setup_griewank()["getdata"]],["test"])

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_plot_objectivefunction(self):
        fig_name = "test_plot_objectivefunction.png"
        spotpy.analyser.plot_objectivefunction(self.results
                                               , self.sampler.evaluation)
        plt.savefig(fig_name)

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_plot_parametertrace_algorithms(self):
        spotpy.analyser.plot_parametertrace_algorithms([self.setup_griewank()["getdata"]],["test_plot_parametertrace_algorithms"])
        fig_name = "test2.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_plot_parametertrace(self):
        spotpy.analyser.plot_parametertrace(self.setup_griewank()["getdata"], ["0","1"])
        fig_name = "0_1__trace.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_plot_posterior_parametertrace(self):
        spotpy.analyser.plot_posterior_parametertrace(self.setup_griewank()["getdata"], ["0","1"])
        fig_name = "0_1__trace.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_plot_posterior(self):
        spotpy.analyser.plot_posterior(self.results[0]
                                       , self.sampler.evaluation)
        fig_name = "bestmodelrun.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_plot_bestmodelrun(self):
        samp = self.setup_griewank()
        spotpy.analyser.plot_bestmodelrun(samp["getdata"], samp["evaluation"])
        fig_name="Best_model_run.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_plot_bestmodelruns(self):
        spotpy.analyser.plot_bestmodelruns(
            self.results, self.sampler.evaluation,
            dates=range(1, 1+len(self.sampler.evaluation)), algorithms=["test"])
        fig_name = "bestmodelrun.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)

    def test_plot_objectivefunctiontraces(self):
        spotpy.analyser.plot_objectivefunctiontraces(self.results
                                                     , self.sampler.evaluation
                                                     , ["test"])
        fig_name="Like_trace.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_plot_regression(self):
        from spotpy.examples.spot_setup_rosenbrock import spot_setup
        spot_setup_object = spot_setup()
        parallel = "mpc"
        dbformat = "ram"
        timeout = 5
        sampler = spotpy.algorithms.mc(spot_setup_object, parallel=parallel,
                                              dbname='test_plot_regression', dbformat=dbformat,
                                              sim_timeout=timeout)
        sampler.sample(300)

        spotpy.analyser.plot_regression(sampler.getdata(), sampler.evaluation)
        fig_name="regressionanalysis.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_plot_parameterInteraction(self):
        self.setup_MC_results()
        spotpy.analyser.plot_parameterInteraction(pickle.load(open("test_analyser_MC_results","rb")))
        fig_name = "ParameterInteraction.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_plot_allmodelruns(self):
        from spotpy.examples.spot_setup_hymod_python import spot_setup as sp
        sp = sp()

        sampler = spotpy.algorithms.dream(sp, parallel="seq", dbname='test_plot_allmodelruns',
                                          dbformat="ram",
                                          sim_timeout=5)

        sampler.sample(50)

        modelruns = []
        for run in sampler.getdata():
            on_run = []
            for i in run:
                on_run.append(i)
            on_run = np.array(on_run)[:-9]
            print(on_run)
            modelruns.append(on_run.tolist())

        test_plot_allmodelruns = spotpy.analyser.plot_allmodelruns(modelruns, sp.evaluation(),
                                                                   dates=range(1, len(sp.evaluation()) + 1))

        fig_name = "bestmodel.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)

        os.remove(fig_name)

    def test_plot_autocorellation(self):
        self.setup_MC_results()

        results = []
        results.append(pickle.load(open("test_analyser_MC_results","rb")))
        results = np.array(results)

        spotpy.analyser.plot_autocorellation(results["parcmax"][0],"parcmax")

        fig_name="Autocorellationparcmax.png"

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the pot just
        # the size
        self.assertGreaterEqual(os.path.getsize(fig_name), 8855)
        os.remove(fig_name)

    def test_plot_gelman_rubin(self):
        from spotpy.examples.spot_setup_hymod_python import spot_setup as sp
        sp = sp()
        sampler = spotpy.algorithms.dream(sp, parallel="seq", dbname='test_plot_autocorellation',
                                          dbformat="csv",
                                          sim_timeout=5)

        r_hat = sampler.sample(100)

        fig_name = "gelman_rubin.png"
        spotpy.analyser.plot_gelman_rubin(r_hat)
        plt.savefig(fig_name)
        self.assertGreaterEqual(abs(os.path.getsize(fig_name)), 100)

        os.remove(fig_name)

    def setup_MC_results(self):

        picklefilename = "test_analyser_MC_results"
        if not os.path.isfile(picklefilename):
            from spotpy.examples.spot_setup_hymod_python import spot_setup as sp
            sp = sp()

            sampler = spotpy.algorithms.mc(sp, parallel="seq", dbname='test_plot_autocorellation',
                                           dbformat="csv",
                                           sim_timeout=5)
            sampler.sample(100)
            pickfil = open(picklefilename, "wb")
            pickle.dump(sampler.getdata(), pickfil)

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove("RosenMC.csv")
            os.remove("setup_griewank_pickle")
            os.remove("test_plot_autocorellation.csv")
            os.remove("test_analyser_MC_results")
            os.remove("Posteriot_parameter_uncertainty.png")
        except FileNotFoundError:
            pass



if __name__ == '__main__':
    unittest.main(exit=False)
