import unittest
import spotpy.signatures as sig
import spotpy
import spotpy.hymod.hymod
import numpy as np
import os

try:
    import pandas as pd
except ImportError:
    print('Please install Pandas to use these signature functions')


#https://docs.python.org/3/library/unittest.html


class spot_setup(object):
    def __init__(self, mean1=-5.0, mean2=5.0, std1=1.0, std2=1.0):

        self.params = [spotpy.parameter.Uniform('x1', low=1.0, high=500, optguess=412.33),
                       spotpy.parameter.Uniform('x2', low=0.1, high=2.0, optguess=0.1725),
                       spotpy.parameter.Uniform('x3', low=0.1, high=0.99, optguess=0.8127),
                       spotpy.parameter.Uniform('x4', low=0.0, high=0.10, optguess=0.0404),
                       spotpy.parameter.Uniform('x5', low=0.1, high=0.99, optguess=0.5592)
                       ]

        self.owd = os.path.dirname(os.path.realpath(__file__)) + os.sep +'..'

        self.evals = list(np.genfromtxt(self.owd + os.sep + 'hymod' + os.sep + 'bound.txt', skip_header=65)[:, 3])[:730]
        self.Factor = 1944 * (1000 * 1000) / (1000 * 60 * 60 * 24)
        #print("That is eval",len(self.evals))

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, x):
        hyMod_sims = spotpy.hymod.hymod.hymod(x[0], x[1], x[2], x[3], x[4])
        simulations = np.array(hyMod_sims) * self.Factor
        # TODO: Discard the first 64 (but it is already done in the script, so what is to do?
        simulations = simulations[66:]
        #print("That is sim", len(simulations))
        return simulations

    def evaluation(self):
        return self.evals

    def objectivefunction(self, simulation, evaluation):


        # like = spotpy.objectivefunctions.log_p(evaluation,simulation)
        # like = spotpy.objectivefunctions.nashsutcliff(evaluation,simulation)-1

        # like = spotpy.likelihoods.NashSutcliffeEfficiencyShapingFactor(evaluation, simulation)
        like = spotpy.objectivefunctions.rmse(evaluation,simulation)
        return like




class TestSignatures(unittest.TestCase):

    def setUp(self):
        self.data = np.random.gamma(0.7,2,500)
        self.spot_setup = spot_setup()
        self.parameterset = self.spot_setup.parameters()['random']
        self.simulation = self.spot_setup.simulation(self.parameterset)
        self.observation = self.spot_setup.evaluation()

        self.timespanlen = self.simulation.__len__()
        try:

            self.ddd = pd.date_range("2015-01-01 11:00", freq="5min", periods=self.timespanlen)
            self.dd_daily = pd.date_range("2015-05-01", periods=self.timespanlen)
            self.usepandas = True
        except NameError:
            print('Please install Pandas to use these signature functions')
            self.usepandas = False

    def test_getSlopeFDC(self):
        sig_val = sig.getSlopeFDC(self.simulation,self.observation, mode="get_signature")
        sig_raw = sig.getSlopeFDC(self.simulation, self.observation, mode="get_raw_data")
        sig_dev = sig.getSlopeFDC(self.simulation, self.observation, mode="calc_Dev")
        self.assertEqual(type(float(sig_val)), type(1.0))
        self.assertEqual(type(float(sig_raw)), type(1.0))
        self.assertEqual(type(float(sig_dev)), type(1.0))
