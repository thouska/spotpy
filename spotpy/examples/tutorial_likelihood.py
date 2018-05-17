# -*- coding: utf-8 -*-
'''
Copyright (c) 2017 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska, Benjamin Manns

This code shows how to use the likelihood framework and present all existing function.
'''

import numpy as np
import spotpy

# First we use all available likelihood functions just alone. The pydoc of every function tells, if we can add a
# parameter `param` to the function which includes model parameter. The `param` must be None or a tuple with values
# and names. If `param` is None, the needed values are calculated by the function itself.

data, comparedata = np.random.normal(1500, 2530, 20), np.random.normal(15, 25, 20)

l = spotpy.likelihoods.logLikelihood(data, comparedata)
print("logLikelihood: " + str(l))

l = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(data, comparedata)
print("gaussianLikelihoodMeasErrorOut: " + str(l))

l = spotpy.likelihoods.gaussianLikelihoodHomoHeteroDataError(data, comparedata)
print("gaussianLikelihoodHomoHeteroDataError: " + str(l))

# Here are examples where functions get `params`
l = spotpy.likelihoods.LikelihoodAR1NoC(data, comparedata, params=([0.98], ["likelihood_phi"]))
print("LikelihoodAR1NoC: " + str(l))

l = spotpy.likelihoods.LikelihoodAR1WithC(data, comparedata)
print("LikelihoodAR1WithC: " + str(l))

l = spotpy.likelihoods.generalizedLikelihoodFunction(data, comparedata, params=
([np.random.uniform(-0.99, 1, 1), np.random.uniform(0.1, 10, 1), np.random.uniform(0, 1, 1), np.random.uniform(0, 1, 0),
  np.random.uniform(0, 0.99, 1), np.random.uniform(0, 100, 1)],
 ["likelihood_beta", "likelihood_xi", "likelihood_sigma0", "likelihood_sigma1", "likelihood_phi1", "likelihood_muh"]))
print("generalizedLikelihoodFunction: " + str(l))

l = spotpy.likelihoods.LaplacianLikelihood(data, comparedata)
print("LaplacianLikelihood: " + str(l))

l = spotpy.likelihoods.SkewedStudentLikelihoodHomoscedastic(data, comparedata)
print("SkewedStudentLikelihoodHomoscedastic: " + str(l))

l = spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedastic(data, comparedata)
print("SkewedStudentLikelihoodHeteroscedastic: " + str(l))

l = spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(data, comparedata, params=(
    [np.random.uniform(2.01, 100, 1), np.random.uniform(0.01, 100, 1), np.random.uniform(-.99, .99, 1)],
    ["likelihood_nu", "likelihood_kappa", "likelihood_phi"]))

print("SkewedStudentLikelihoodHeteroscedasticAdvancedARModel: " + str(l))

l = spotpy.likelihoods.NoisyABCGaussianLikelihood(data, comparedata)
print("NoisyABCGaussianLikelihood: " + str(l))

l = spotpy.likelihoods.ABCBoxcarLikelihood(data, comparedata)
print("ABCBoxcarLikelihood: " + str(l))

l = spotpy.likelihoods.LimitsOfAcceptability(data, comparedata)
print("LimitsOfAcceptability: " + str(l))

l = spotpy.likelihoods.InverseErrorVarianceShapingFactor(data, comparedata)
print("inverseErrorVarianceShapingFactor: " + str(l))

l = spotpy.likelihoods.ExponentialTransformErrVarShapingFactor(data, comparedata)
print("inverseErrorVarianceShapingFactor: " + str(l))

l = spotpy.likelihoods.NashSutcliffeEfficiencyShapingFactor(data, comparedata)
print("NashSutcliffeEfficiencyShapingFactor: " + str(l))

l = spotpy.likelihoods.sumOfAbsoluteErrorResiduals(data, comparedata)
print("sumOfAbsoluteErrorResiduals: " + str(l))


# We also can use the likelihood functions in an algorithmus. We will need a setup class like this


class spot_setup_gauss(object):
    def __init__(self):
        self.params = [
            # Original mean: 12, sd:23
            spotpy.parameter.Uniform('mean', -20, 20, 2, 3.0, -20, 20),
            spotpy.parameter.Uniform('sd', 1, 30, 2, 3.01, 1, 30),

            # Some likelihood function need additional parameter, look them up in the documentation
            # spotpy.parameter.Uniform('likelihood_nu', 2.01, 100, 1.5, 3.0, -10, 10),
            # spotpy.parameter.Uniform('likelihood_kappa', 0.01, 100, 1.5, 3.0, -10, 10),
            #spotpy.parameter.Uniform('likelihood_phi', -.99, .99, 0.1, 0.1, 0.1, 0.1),

            # spotpy.parameter.Uniform('likelihood_beta', -.99, .99, 1.5, 3.0, -10, 10),
            # spotpy.parameter.Uniform('likelihood_xsi', 0.11, 10, 1.5, 3.0, -10, 10),
            # spotpy.parameter.Uniform('likelihood_sigma0', 0, 1, 1.5, 3.0, -10, 10),

            # spotpy.parameter.Uniform('likelihood_sigma1', 0, 1, 1.5, 3.0, -10, 10),
            # spotpy.parameter.Uniform('likelihood_phi1', 0, .99, 1.5, 3.0, -10, 10),
            # spotpy.parameter.Uniform('likelihood_muh', 0, 100, 1.5, 3.0, -10, 10)

        ]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        # x = np.random.randint(-100, 100, size=201)
        # simulations= [sum(100.0 * (x[1:] - x[:-1] **2.0) **2.0 + (1 - x[:-1]) **2.0)]
        import subprocess

        import json

        output = subprocess.check_output(
            "/Library/Frameworks/R.framework/Versions/3.1/Resources/Rscript /Users/Karlson/Documents/03_PRIVAT/Privat_UNI/Uni_JOB/Job_AGRAR/develop/spotpy/spotpy/likelihood_test/myR_2.R " + str(
                vector[0])+" "+str(vector[1]) , shell=True)
        #print("this parameters: "+str(vector[0])+" "+str(vector[1]))

        output = output.decode("utf-8")
        back = json.loads(output)

        simulations = back

        return simulations

    def evaluation(self):

        # Gauss with  mean=12 and sd=23
        observations = [1.537678, 7.615278, 33.54329, 12.09963, 24.69595, 7.033905, 21.30595, -17.77526, -7.09708,
                        36.80745, 10.68426, -42.7048, -21.01126, -6.314566, 38.01058, -15.79536, -17.69119, -4.482229,
                        12.30351, -30.54512, 8.468925, 27.44369, 37.20623, -8.753253, 39.40037, 29.03273, 0.5257918,
                        25.98343, -16.09876, 6.430084, 4.755722, -10.38204, 7.97673, -37.55442, 58.04988, 20.41361,
                        32.13943, 30.37884, 6.898094, 13.32948, -14.5311, 25.0606, 25.81364, 25.82836, 47.70208,
                        31.1919, 24.74743, 18.21143, 10.67086, 47.29963, 40.3718, 39.21012, 6.774497, -4.244702,
                        13.45878, 33.80645, 15.64674, -6.277918, 36.83417, 23.13524, 11.85227, 31.38894, 13.00289,
                        6.47117, 19.31257, -11.23233, 21.07677, 14.96549, 9.952789, 23.54167, 46.7991, 47.64822,
                        7.33875, 17.64916, 38.79842, 11.75935, 17.70734, 15.64669, -6.890646, 3.710825, 44.42125,
                        -20.1855, 24.32393, 56.55909, 33.02915, 5.173076, -24.00348, 16.62815, -21.64049, 18.2159,
                        41.69109, -31.26055, 36.9492, 2.780838, -4.519057, -24.71357, 20.63503, 17.08391, 26.23503,
                        12.82442, 22.13652, 21.21188, 47.99579, 44.52914, -0.5511025, 55.47107, -15.12694, 2.884632,
                        7.361032, 12.66143, 37.38807, 53.63648, 9.114074, 12.68311, -6.890102, 32.40405, 22.93079,
                        1.498509, 22.68785, 29.71565, 21.42051, -9.961459, -10.22352, -28.16017, 14.14882, 9.64758,
                        -5.821728, -21.93086, 19.94631, 16.29195, 28.87528, 25.81239, 52.44341, 5.229822, -17.92572,
                        11.85504, 17.21691, 17.19854, -6.37061, 16.1524, -13.08297, 13.45471, 9.43481, -2.177022,
                        17.46416, 2.647446, 16.77834, 20.77827, 49.37553, 8.435563, -13.85352, 17.06572, 5.550149,
                        2.674943, -21.95848, 11.82037, 30.51478, 8.334891, -17.1576, -16.82652, 50.31279, -31.05799,
                        52.69635, 22.11049, -43.32149, -13.5348, -5.125771, 1.801732, 19.30368, 14.94216, -19.32855,
                        20.75345, 21.03398, 5.430688, -3.163607, 10.1321, -12.9058, -13.77746, 25.02506, -7.187246,
                        39.93147, 5.330449, 6.705344, -16.47149, -34.20934, 28.66261, -6.420032, 4.682751, -9.622858,
                        17.95173, 3.316576, 14.6763, 13.84716, -20.52151, 24.35037, 28.57057, 18.17082, 26.14141,
                        72.05923, -12.29775,26.83472]

        return observations

    def objectivefunction(self, simulation=simulation, evaluation=evaluation, params=None):
        # Some functions do not need a `param` attribute, you will see that in the documentation or if an error occur.
        # objectivefunction = spotpy.likelihoods.LimitsOfAcceptability(evaluation, simulation,params=params)
        #objectivefunction = spotpy.likelihoods.NoisyABCGaussianLikelihood(evaluation, simulation)
        #objectivefunction = spotpy.likelihoods.LimitsOfAcceptability(evaluation, simulation)
        objectivefunction = spotpy.objectivefunctions.rmse(simulation=simulation,evaluation=evaluation)
        #print(objectivefunction)

        return objectivefunction

class spot_setup_wald(object):
    def __init__(self):
        self.params = [
            # See https://de.wikipedia.org/wiki/Inverse_Normalverteilung
            # Original 12,23
            spotpy.parameter.Uniform('lambda', 1, 1000, 0.1, 3.0, -20, 20),
            #spotpy.parameter.Uniform('nu', 1, 30, 2, 3.01, 1, 30),
]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        import subprocess

        import json

        output = subprocess.check_output(
            "/Library/Frameworks/R.framework/Versions/3.1/Resources/Rscript /Users/Karlson/Documents/03_PRIVAT/Privat_UNI/Uni_JOB/Job_AGRAR/develop/spotpy/spotpy/likelihood_test/myR_WALD.R " + str(vector[0]), shell=True)
        # print("this parameters: "+str(vector[0])+" "+str(vector[1]))

        output = output.decode("utf-8")
        back = json.loads(output)

        simulations = back

        return simulations

    def evaluation(self):
        # Wald distrubtion with lambda=42 (nu = 1 as it is original inverse Gauß)
        observations = [0.8215101,1.050744,1.068614,0.9237615,1.134586,0.8342905,0.9203649,0.8423139,1.016296,0.819583,0.7727125,1.049373,0.9064652,1.129859,0.7390692,0.7807588,0.9512094,0.751157,0.8342608,0.9535379,0.8855571,0.8164966,0.9859118,0.9663425,0.9168434,1.096442,1.075291,0.7939873,0.8371087,0.8899696,0.8223036,0.9441274,1.251677,0.9946841,0.9688333,0.8584872,1.118507,1.092399,0.9389445,1.320034,1.05912,0.8073291,0.9718409,0.9993603,1.009801,1.191749,1.096261,0.9104541,1.135112,1.024141,0.68865,1.117724,1.071344,0.9730503,1.03879,0.9040554,1.226641,1.090904,0.9188659,0.9516232,1.111537,0.7868174,1.03979,0.8529991,1.546705,0.9973017,0.9056773,1.020306,0.8666091,0.8227436,1.107373,1.240635,0.8642053,1.012499,0.8189009,0.9112955,0.9133874,0.764895,0.9954879,1.016124,1.135945,1.210386,0.8935554,1.133396,0.8744451,1.27583,0.9399524,1.109192,1.024147,1.010473,0.823447,1.063746,1.587057,1.25963,1.075372,0.9058057,1.149925,0.8951753,0.8786255,0.7846421,1.089261,1.155204,0.9162714,1.091259,1.012252,0.9860885,1.000627,0.8831002,0.9736084,1.020061,1.099529,0.8503705,1.092692,1.018956,0.9597126,0.9760877,0.8305396,1.010439,0.8659965,1.216233,1.15933,0.8352535,1.086288,1.085791,1.215822,1.455505,0.8928623,1.227453,0.9778177,1.248284,0.678555,0.9379088,1.076307,1.081512,1.056103,0.9475012,1.11073,1.216543,1.409434,0.8973831,0.7879291,1.039925,0.9954887,0.8129037,1.088005,1.010168,0.9842995,1.034264,0.9122271,1.128363,1.331232,1.206762,1.134155,1.166505,1.154047,1.054108,1.07734,0.8397705,0.9748741,1.133629,0.9498966,0.9889976,1.023417,0.9424091,0.9424539,1.246194,0.9413468,1.15047,0.7332654,1.496362,0.828069,0.7696388,0.918564,0.8388578,0.9455839,0.8227491,0.9551339,0.963993,1.051606,1.013032,0.8144458,1.07049,1.029506,0.7699333,0.9409208,1.341655,1.023382,0.9868176,0.9950876,0.954334,0.957515,1.136036,1.265562,0.9722909,0.7632513,0.8805661,0.8904488,1.052702,1.036818,0.9569595,0.9428334]

        return observations

    def objectivefunction(self, simulation=simulation, evaluation=evaluation, params=None):
        objectivefunction = spotpy.likelihoods.NoisyABCGaussianLikelihood(evaluation, simulation)

        return objectivefunction

class spot_setup_ar_1_students_t_res(object):
    def __init__(self):
        self.params = [
            # For students - Skew
            spotpy.parameter.Uniform('likelihood_kappa', 1, 1, 1, 1, 1, 1),
            spotpy.parameter.Uniform('likelihood_phi', -0.99, 0.99, 0.1, 3.0, -0.99, 0.99),
            spotpy.parameter.Uniform('likelihood_nu', 2.1, 100, 2, 3.01, 2.1, 10),

        ]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        import subprocess

        import json
        # parameter 0:phi, 1:nu
        output = subprocess.check_output(
            "/Library/Frameworks/R.framework/Versions/3.1/Resources/Rscript /Users/Karlson/Documents/03_PRIVAT/Privat_UNI/Uni_JOB/Job_AGRAR/develop/spotpy/spotpy/likelihood_test/myR_AR1_Student-t-res.R " + str(
                vector[0])+" "+str(vector[1]), shell=True)
        #print("this parameters: "+str(vector[0])+" "+str(vector[1]) + " "+str(vector[2]))

        output = output.decode("utf-8")
        back = json.loads(output)

        simulations = back

        return simulations

    def evaluation(self):
        # AR1 with student-t distributed residuals with phi=42 and df=7
        observations = [-0.806554,-1.036565,-1.054924,1.207057,-0.2267287,-0.6241089,-0.9865154,0.004914617,0.5435943,-0.2529109,-0.220569,-1.085152,-1.446932,-0.2271839,-0.9695579,-1.042956,-1.677816,-0.3111386,0.2023657,1.681282,0.3311756,-2.011093,-0.05581758,1.792977,1.469434,-0.4426333,0.8043387,1.093684,1.99667,-0.1171755,0.5409204,-1.036451,-0.03801139,0.6125889,0.09342139,1.977853,0.5931005,2.947713,-1.51345,0.152891,-1.079158,-1.466529,-1.504046,2.181459,-1.966612,0.3770235,-0.7682167,-0.2559416,1.215604,0.9032672,0.7817497,-1.634914,-0.1532924,-1.329244,-3.425294,-0.9132035,-0.1814598,-0.3703962,0.8464317,0.3035075,-0.1755196,-0.7057517,-0.6808711,-1.221144,0.04788332,0.7737842,0.9854636,0.7845628,3.702611,3.388055,1.097186,0.672756,-0.3338004,0.2702849,-0.6202132,-2.228129,-0.7818483,0.155476,0.657133,0.5652424,0.0008205475,0.4045611,0.7796379,0.4325643,-0.354133,-1.166525,-0.9882422,0.4509809,-0.4848221,0.1715434,-1.196337,0.4085583,-0.04952907,1.035795,0.9496666,0.3619785,0.8666738,0.285121,-1.432747,-1.939233,-0.1684185,0.003022823,-0.4922612,0.5042833,0.09296572,-1.455734,-0.213669,1.063985,0.2004293,-1.649053,0.200737,-0.105006,-2.633298,-3.161435,-0.5550424,-1.713277,-0.5169007,-0.3834445,-0.2791567,1.262414,0.06332764,1.640421,0.3144967,-0.01392552,-1.151283,-1.268196,-1.880887,1.010745,0.2109186,2.752468,0.09242852,-2.878202,-1.477239,-2.243395,-0.6384147,-0.6950003,-1.686313,-0.6945071,1.603874,0.09509077,-0.09816484,1.150218,2.324576,1.504623,-1.063881,1.330131,2.611619,2.494136,0.6338899,1.004135,0.1381033,-0.360101,1.124663,1.78177,1.603268,1.325218,2.4939,2.153747,1.97724,1.18353,-0.7253327,0.823499,0.4020309,0.6603788,0.129381,-1.321761,1.939023,0.6186619,0.9888297,1.118437,0.2581724,1.385318,1.997055,-0.8354623,0.441386,-0.5992372,-2.595201,1.591507,-0.7960406,0.2250709,0.6408333,-0.06189073,0.103896,-0.4334433,0.1433027,-0.6207327,0.2989117,-0.2577758,-0.5267562,-3.398514,0.1546277,1.012887,0.06408349,1.258883,0.5752389,-1.694637,-0.7765886,2.41088,0.2273703,-1.2156,-1.232374]

        return observations

    def objectivefunction(self, simulation=simulation, evaluation=evaluation, params=None):
        objectivefunction = spotpy.likelihoods.SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(evaluation, simulation,params=params)
        if np.isinf(objectivefunction):
            objectivefunction = 0.0
        print("Class log "+str(objectivefunction))
        return objectivefunction



class spot_setup_ar_1_gauss_res(object):
    def __init__(self):
        self.params = [
            spotpy.parameter.Uniform('likelihood_phi', -0.99, 0.99, 0.2, 3.0, -0.99, 0.99),
        ]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        import subprocess

        import json
        # parameter 0:phi
        output = subprocess.check_output(
            "/Library/Frameworks/R.framework/Versions/3.1/Resources/Rscript /Users/Karlson/Documents/03_PRIVAT/Privat_UNI/Uni_JOB/Job_AGRAR/develop/spotpy/spotpy/likelihood_test/myR_AR1_Norm_res.R " + str(
                vector[0]), shell=True)
        print("this parameters: "+str(vector[0]))

        output = output.decode("utf-8")
        back = json.loads(output)

        simulations = back

        return simulations

    def evaluation(self):
        # AR1 with normal distributed residuals with phi=0.428 and mean=0,sd=1
        observations = [-2.522743,1.408725,1.946646,0.888204,0.7897667,0.6112302,2.290427,2.868624,1.556995,0.05254887,0.7461225,0.3719867,0.979471,-1.190781,0.4436523,2.236696,0.2605191,0.673389,0.7251472,0.6608128,-0.7443824,0.371268,0.02141081,0.4607711,-0.5507639,-1.005326,0.7367659,0.5269135,0.1166365,-0.5970824,-0.7521213,0.367761,0.1125972,-0.0795236,0.1165288,0.6486639,1.932282,1.033983,-0.9876154,0.1796362,-0.3983238,0.01101198,0.9050182,-0.6080171,0.08849208,-1.81438,-0.07913209,-0.009007132,-1.160989,-1.297887,1.419709,-1.066902,-1.270009,-1.030414,-2.38863,-0.2904009,0.7489411,-0.1846965,-1.433198,-1.145359,-0.04856701,1.087533,0.7987545,0.641762,-0.378111,-1.019192,0.2837018,2.259854,0.4158938,1.451425,0.9710148,1.325311,0.04831358,-0.2373003,0.09663009,-2.557514,-0.9230433,-0.6250428,-0.6359625,-0.2501693,1.096061,1.564296,1.956924,-0.4511307,-0.521957,-0.3384552,-0.3905848,-1.05168,1.266555,1.835193,1.168423,-0.5542428,-0.7080179,0.8867539,0.9273763,0.9104679,1.761588,2.650327,0.549592,1.152438,0.1226201,-0.1466213,0.6201685,-0.244967,1.728424,-0.1991486,0.4715499,2.541504,2.055189,0.5441279,1.075167,0.6381296,0.7177508,1.106246,-0.6729018,-2.086599,-1.199925,0.7879157,0.01633025,-0.5845763,0.4181293,-2.16246,-2.197336,-2.209761,-1.856266,-1.021362,0.6899624,-0.898831,0.3702167,1.202066,0.5307163,-1.183152,-1.305882,-1.34188,0.9997525,-1.611335,0.4367853,1.24461,2.474933,0.9325948,2.022836,-0.6440014,-1.253051,0.2687869,-1.139736,0.1336643,-1.383847,0.3793314,-0.7788819,0.8646879,1.433118,1.026648,1.31162,-0.8718095,0.3493136,-0.2196879,0.3182434,-0.7072177,0.6959091,-0.5620885,0.8712038,0.6010974,-0.007788187,-0.1008797,-0.5404524,-0.6134115,2.287364,-0.04623299,-1.30409,0.6175855,-1.475111,0.2202588,0.5428336,0.3996769,0.1243578,0.3912388,1.346204,0.6947638,-0.5677999,0.3445474,-0.7952659,1.144519,1.850942,-0.312677,-1.347769,-0.6379291,0.1777509,0.1736661,-0.6718341,0.8210482,-0.4401946,0.9218458,-0.2263964,0.2356263,-0.6045727,0.124017,0.6440486,1.095587,0.4844389,-0.4179212,1.385794]

        return observations

    def objectivefunction(self, simulation=simulation, evaluation=evaluation, params=None):
        objectivefunction = spotpy.likelihoods.LikelihoodAR1NoC(evaluation, simulation,params=params)
        
        return objectivefunction




class spot_setup_generalizedGauss(object):
    def __init__(self):
        self.params = [

            spotpy.parameter.Uniform('likelihood_phi1', 0.01, 0.99, 3.0, 3.0, 0.01, 0.99),
            spotpy.parameter.Uniform('likelihood_xi', 0.1, 10, 3, 3, 0.1, 10),
            spotpy.parameter.Uniform('likelihood_beta', 0.1, 0.99, 3, 3.0, -0.99, 0.99), # positive - simulation in R says it!


            spotpy.parameter.Uniform('likelihood_sigma0', 0, 1, 0.1, 3.0, 0, 1),
            spotpy.parameter.Uniform('likelihood_sigma1', 0, 1, 0.1, 3.0, 0, 1),

            spotpy.parameter.Uniform('likelihood_muh', 0, 100, 0.1, 3.0, 0, 100),
        ]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        import subprocess

        import json
        # we need a skew exponential power to add xi and beta in it which is the model for the error residuals of AR1 Model

        output = subprocess.check_output(
            "/Library/Frameworks/R.framework/Versions/3.1/Resources/Rscript /Users/Karlson/Documents/03_PRIVAT/Privat_UNI/Uni_JOB/Job_AGRAR/develop/spotpy/spotpy/likelihood_test/myR_AR1_SEP_res.R " + str(
                vector[0]) + " " + str(vector[1])+ " "+str(vector[2]), shell=True)
        # print("this parameters: "+str(vector[0])+" "+str(vector[1]) + " "+str(vector[2]))

        output = output.decode("utf-8")
        back = json.loads(output)
        if back[0] == "NA":
            back = np.repeat(np.NAN,back.__len__()).tolist()
        simulations = back

        return simulations

    def evaluation(self):
        # AR1 with SEP distribution phi=0.142 xsi=3, beta=9.5
        # [  0.99         1.28841626   0.99         0.43366888   0.38087079 72.72585542]
        observations = [1.367689,-0.320067,-0.04381581,0.6023338,0.9038274,1.034441,-0.3475758,-0.6766884,0.375266,0.3902351,1.41773,1.146159,1.142029,-0.5467857,1.132456,0.05771065,-0.01329709,1.245674,1.262945,0.5637976,-0.6106627,-0.1347206,0.4439383,-0.1191365,0.6781304,-0.4293178,0.1856715,0.4008803,1.34489,0.9124905,1.237749,0.5098399,0.8364595,-0.4464507,0.6631744,0.2039722,-0.05081068,0.7299973,0.8854515,1.180466,-0.4876658,0.7830223,-0.4316994,1.099141,0.5340687,0.8495034,0.8779076,0.6735508,-0.3102846,0.2900948,-0.05825545,-0.941212,1.025862,0.7281562,-0.361788,0.6388547,1.038316,1.343974,0.8034503,1.39158,0.5718842,0.4621339,0.828369,1.091347,0.9504174,1.100152,0.4411185,1.178236,1.528249,1.311347,1.011896,0.6851925,1.102152,1.191884,-0.3198258,1.023772,1.021118,0.351345,-0.7778747,0.3130401,0.8374449,-0.2165474,0.6511311,0.9294736,1.007714,1.124968,1.122693,1.053253,1.064944,-0.3810931,-0.7520672,0.07500417,-0.6589652,0.9858736,-0.3338579,0.5976432,1.065922,-0.1717056,-0.7935736,-0.2154963,0.1094597,0.9271599,0.8882699,-0.204626,-0.1153957,-0.03153619,1.145353,0.1135476,-0.0652023,-0.3510398,-0.8471455,-0.7796421,-0.2307928,-0.2594656,0.8092929,0.4113968,-0.188539,1.19418,0.8070983,0.9118222,1.071649,1.051291,-0.8035766,0.8092788,-0.1163294,-0.02733921,-0.6852544,-0.408636,-0.08736997,0.4578535,0.7799243,0.4268271,0.5604364,-0.1452668,-0.1654945,-0.483941,0.1326935,0.4563893,0.7192259,0.7154398,1.120117,0.3798121,-0.1878339,-0.5358535,-0.5510031,0.0233894,0.07423701,0.9234318,-0.1600513,1.372747,0.6790618,0.8772782,-0.1664986,0.9622479,0.9873338,0.2296364,-1.002397,-0.2306121,-0.1446204,0.31194,-0.1342179,0.08709208,-0.2634807,-0.5770442,0.5588156,-0.4229277,-0.8920537,-0.3130578,0.9963966,-0.1462349,0.2177117,1.019684,0.3005968,0.7721734,-0.1104061,-0.7366346,0.9962065,0.4035172,1.175904,1.103031,-0.742134,-0.3189378,0.5614889,0.4403444,-0.6407969,-0.4805289,0.2666954,0.8946856,0.200042,0.700875,-0.3121022,-0.4439033,-0.7696692,0.6370263,1.367421,0.8487393,0.1497969,-0.4690384,1.088799,0.0210073,0.3703306]

        return observations

    def objectivefunction(self, simulation=simulation, evaluation=evaluation, params=None):
        objectivefunction = spotpy.likelihoods.generalizedLikelihoodFunction(evaluation,simulation,params=params)
        if np.isinf(objectivefunction):
            objectivefunction = 0.0
        print("Class log " + str(objectivefunction))
        return objectivefunction




# And now we can start an algorithm to find the best possible data


results = []
spot_setup = spot_setup_gauss()
rep = 1000

# TODO alt_objfun immer wieder anschalten sonst Trick 17!
sampler = spotpy.algorithms.lhs(spot_setup, dbname='RosenMC', dbformat='csv', alt_objfun=None)
sampler.sample(rep)
results.append(sampler.getdata())
import pprint
import pickle

pickle.dump(results, open('mle_LimitsOfAcceptability_v2.p', 'wb'))

# from matplotlib import pyplot as plt
# plt.plot(results)
# plt.show()
# plt.plot(results['like1'])
# plt.show()
