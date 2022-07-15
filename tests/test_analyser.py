# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska, Benjamin Manns
"""


import matplotlib as mpl

mpl.use("Agg")
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

import spotpy
import spotpy.analyser
from spotpy.examples.spot_setup_griewank import spot_setup as griewank_setup
from spotpy.examples.spot_setup_hymod_python import spot_setup as hymod_setup
from spotpy.examples.spot_setup_rosenbrock import spot_setup as rosenbrock_setup
from spotpy.likelihoods import gaussianLikelihoodMeasErrorOut as GausianLike


class TestAnalyser(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(42)
        self.rep = 300
        self.parallel = "seq"
        self.dbformat = "ram"
        self.timeout = 5
        self.fig_name = "test_output.png"

        sampler = spotpy.algorithms.mc(rosenbrock_setup(), sim_timeout=self.timeout)
        sampler.sample(self.rep)
        self.results = sampler.getdata()

        sampler = spotpy.algorithms.mc(griewank_setup(), sim_timeout=self.timeout)
        sampler.sample(self.rep)
        self.griewank_results = sampler.getdata()

        sampler = spotpy.algorithms.fast(rosenbrock_setup(), sim_timeout=self.timeout)
        sampler.sample(self.rep)
        self.sens_results = sampler.getdata()
        # Hymod resuts are empty with Python <3.6
        sampler = spotpy.algorithms.dream(
            hymod_setup(GausianLike), sim_timeout=self.timeout
        )
        self.r_hat = sampler.sample(self.rep)
        self.hymod_results = sampler.getdata()

    def test_setUp(self):
        self.assertEqual(len(self.results), self.rep)
        self.assertEqual(len(self.griewank_results), self.rep)
        self.assertEqual(len(self.hymod_results), self.rep)
        self.assertEqual(len(self.sens_results), self.rep)

    def test_get_parameters(self):

        self.assertEqual(len(spotpy.analyser.get_parameters(self.results)[0]), 3)

    def test_get_parameternames(self):
        self.assertEqual(
            spotpy.analyser.get_parameternames(self.results), ["x", "y", "z"]
        )

    def test_get_parameter_fields(self):
        self.assertEqual(len(spotpy.analyser.get_parameternames(self.results)), 3)

    def test_get_minlikeindex(self):
        minlikeindex = spotpy.analyser.get_minlikeindex(self.results)
        self.assertEqual(len(minlikeindex), 2)
        self.assertEqual(type(minlikeindex), type((1, 1)))

    def test_get_maxlikeindex(self):
        get_maxlikeindex = spotpy.analyser.get_maxlikeindex(self.results)
        self.assertEqual(len(get_maxlikeindex), 2)
        self.assertEqual(type(get_maxlikeindex), type((1, 1)))

    def test_get_like_fields(self):
        get_like_fields = spotpy.analyser.get_like_fields(self.results)
        self.assertEqual(len(get_like_fields), 1)
        self.assertEqual(type(get_like_fields), type([]))

    def test_calc_like(self):
        calc_like = spotpy.analyser.calc_like(
            self.results,
            rosenbrock_setup().evaluation(),
            spotpy.objectivefunctions.rmse,
        )
        self.assertEqual(len(calc_like), len(self.results))
        self.assertEqual(type(calc_like), type([]))

    def test_get_best_parameterset(self):
        get_best_parameterset_true = spotpy.analyser.get_best_parameterset(
            self.results, True
        )
        get_best_parameterset_false = spotpy.analyser.get_best_parameterset(
            self.results, False
        )
        self.assertEqual(len(get_best_parameterset_true[0]), 3)
        self.assertEqual(type(get_best_parameterset_true[0]), np.void)
        self.assertEqual(len(get_best_parameterset_false[0]), 3)
        self.assertEqual(type(get_best_parameterset_false[0]), np.void)

    def test_get_modelruns(self):
        get_modelruns = spotpy.analyser.get_modelruns(self.results)
        self.assertEqual(len(get_modelruns[0]), 1)
        self.assertEqual(type(get_modelruns[0]), np.void)

    def test_get_header(self):
        get_header = spotpy.analyser.get_header(self.results)
        self.assertEqual(len(get_header), 6)
        self.assertEqual(type(get_header), type(()))

    def test_get_min_max(self):
        get_min_max = spotpy.analyser.get_min_max(spotpy_setup=rosenbrock_setup())
        self.assertEqual(len(get_min_max[0]), 3)
        self.assertEqual(type(get_min_max), type(()))

    def test_get_parbounds(self):
        get_parbounds = spotpy.analyser.get_parbounds(spotpy_setup=rosenbrock_setup())
        self.assertEqual(len(get_parbounds[0]), 2)
        self.assertEqual(len(get_parbounds), 3)
        self.assertEqual(type(get_parbounds), type([]))

    def test_get_percentiles(self):
        get_percentiles = spotpy.analyser.get_percentiles(self.results)
        self.assertEqual(len(get_percentiles), 5)
        self.assertEqual(type(get_percentiles[0][0]), type(np.abs(-1.0)))
        self.assertEqual(type(get_percentiles), type(()))

    def test__geweke(self):
        sample1 = []
        for a in self.results:
            sample1.append(a[0])

        _geweke = spotpy.analyser._Geweke(sample1)

        self.assertEqual(len(_geweke), 20)
        self.assertEqual(type(_geweke), type(np.array([])))

    def test_plot_Geweke(self):
        sample1 = []
        for a in self.results:
            sample1.append(a[0])
        spotpy.analyser.plot_Geweke(sample1, "sample1")
        plt.savefig(self.fig_name)

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_get_sensitivity_of_fast(self):
        get_sensitivity_of_fast = spotpy.analyser.get_sensitivity_of_fast(
            self.sens_results
        )
        self.assertEqual(len(get_sensitivity_of_fast), 2)
        self.assertEqual(len(get_sensitivity_of_fast["S1"]), 3)
        self.assertEqual(len(get_sensitivity_of_fast["ST"]), 3)
        self.assertEqual(type(get_sensitivity_of_fast), type({}))

    def test_get_simulation_fields(self):
        get_simulation_fields = spotpy.analyser.get_simulation_fields(self.results)
        self.assertEqual(["simulation_0"], get_simulation_fields)

    def test_compare_different_objectivefunctions(self):

        sampler = spotpy.algorithms.lhs(rosenbrock_setup(), sim_timeout=self.timeout)

        sampler.sample(self.rep)
        compare_different_objectivefunctions = (
            spotpy.analyser.compare_different_objectivefunctions(
                sampler.getdata()["like1"], self.results["like1"]
            )
        )

        self.assertEqual(
            type(compare_different_objectivefunctions[1]), type(np.array([0.5])[0])
        )

    def test_plot_parameter_uncertainty(self):
        posterior = spotpy.analyser.get_posterior(self.hymod_results, percentage=10)
        # assertAlmostEqual tests on after comma accuracy, therefor we divide both by 100
        self.assertAlmostEqual(len(posterior) / 100, self.rep * 0.001, 1)
        self.assertEqual(type(posterior), type(np.array([])))
        spotpy.analyser.plot_parameter_uncertainty(
            posterior, hymod_setup().evaluation(), fig_name=self.fig_name
        )

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_plot_fast_sensitivity(self):

        spotpy.analyser.plot_fast_sensitivity(self.sens_results, fig_name=self.fig_name)
        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_plot_heatmap_griewank(self):
        spotpy.analyser.plot_heatmap_griewank(
            [self.griewank_results], ["test"], fig_name=self.fig_name
        )

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_plot_objectivefunction(self):
        spotpy.analyser.plot_objectivefunction(
            self.results, rosenbrock_setup().evaluation(), fig_name=self.fig_name
        )

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_plot_parametertrace_algorithms(self):
        spotpy.analyser.plot_parametertrace_algorithms(
            [self.griewank_results],
            ["test_plot_parametertrace_algorithms"],
            spot_setup=griewank_setup(),
            fig_name=self.fig_name,
        )
        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)
        os.remove(self.fig_name)

    def test_plot_parametertrace(self):
        spotpy.analyser.plot_parametertrace(
            self.griewank_results, ["0", "1"], fig_name=self.fig_name
        )

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_plot_posterior_parametertrace(self):
        spotpy.analyser.plot_posterior_parametertrace(
            self.griewank_results, ["0", "1"], fig_name=self.fig_name
        )

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_plot_posterior(self):
        spotpy.analyser.plot_posterior(
            self.hymod_results, hymod_setup().evaluation(), fig_name=self.fig_name
        )

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_plot_bestmodelrun(self):
        spotpy.analyser.plot_bestmodelrun(
            self.griewank_results, griewank_setup().evaluation(), fig_name=self.fig_name
        )

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)
        os.remove(self.fig_name)

    def test_plot_bestmodelruns(self):
        spotpy.analyser.plot_bestmodelruns(
            [self.hymod_results[0:10], self.hymod_results[10:20]],
            hymod_setup().evaluation(),
            dates=range(1, 1 + len(hymod_setup().evaluation())),
            algorithms=["test", "test2"],
            fig_name=self.fig_name,
        )

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_plot_objectivefunctiontraces(self):
        spotpy.analyser.plot_objectivefunctiontraces(
            [self.results],
            [rosenbrock_setup().evaluation()],
            ["test"],
            fig_name=self.fig_name,
        )

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6820)

    def test_plot_regression(self):
        spotpy.analyser.plot_regression(
            self.results, rosenbrock_setup().evaluation(), fig_name=self.fig_name
        )

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_plot_parameterInteraction(self):
        # Test only untder Python 3.6 as lower versions results in a strange ValueError
        spotpy.analyser.plot_parameterInteraction(self.results, fig_name=self.fig_name)

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_plot_allmodelruns(self):
        modelruns = []
        for run in self.hymod_results:
            on_run = []
            for i in run:
                on_run.append(i)
            on_run = np.array(on_run)[:-7]
            modelruns.append(on_run.tolist())
        spotpy.analyser.plot_allmodelruns(
            modelruns,
            hymod_setup().evaluation(),
            dates=range(1, len(hymod_setup().evaluation()) + 1),
            fig_name=self.fig_name,
        )

        # approximately 8855 KB is the size of an empty matplotlib.pyplot.plot, so
        # we expecting a plot with some content without testing the structure of the plot, just
        # the size
        self.assertGreaterEqual(os.path.getsize(self.fig_name), 6855)

    def test_plot_gelman_rubin(self):
        spotpy.analyser.plot_gelman_rubin(
            self.hymod_results, self.r_hat, fig_name=self.fig_name
        )
        self.assertGreaterEqual(abs(os.path.getsize(self.fig_name)), 100)

    @classmethod
    def tearDownClass(self):
        os.remove("test_output.png")


if __name__ == "__main__":
    unittest.main(exit=False)
