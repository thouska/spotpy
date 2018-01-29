# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska

This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This tool holds functions for statistic analysis. It takes Python-lists and
returns the objective function value of interest.
'''

import numpy as np
import logging
logging.basicConfig(format='%(levelname)s: %(module)s.%(funcName)s(): %(message)s')

def bias(evaluation, simulation):
    """
    Bias as shown in Gupta in Sorooshian (1998), Toward improved calibration of hydrologic models: 
    Multiple  and noncommensurable measures of information, Water Resources Research

        .. math::

         Bias=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Bias
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        bias_values = []
        for i in range(len(evaluation)):
            bias_values.append(float(evaluation[i]) - float(simulation[i]))
        bias_sum = np.sum(bias_values[0:len(bias_values)])
        bias = bias_sum / len(bias_values)
        return float(bias)

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def pbias(evaluation, simulation):
    """
    Procentual Bias

        .. math::

         PBias= 100 * \\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})}{\\sum_{i=1}^{N}(e_{i})}

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: PBias
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        sim = np.array(simulation)
        obs = np.array(evaluation)
        return 100 * (float(np.sum(sim - obs)) / float(np.sum(obs)))

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def nashsutcliffe(evaluation, simulation):
    """
    Nash-Sutcliffe model efficinecy

        .. math::

         NSE = 1-\\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2} 

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Nash-Sutcliff model efficiency
    :rtype: float

    """
    if len(evaluation) == len(simulation):
        s, e = np.array(simulation), np.array(evaluation)
        # s,e=simulation,evaluation
        mean_observed = np.mean(e)
        # compute numerator and denominator
        numerator = sum((e - s) ** 2)
        denominator = sum((e - mean_observed)**2)
        # compute coefficient
        return 1 - (numerator / denominator)

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def lognashsutcliffe(evaluation, simulation):
    """
    log Nash-Sutcliffe model efficiency

        .. math::

         NSE = 1-\\frac{\\sum_{i=1}^{N}(log(e_{i})-log(s_{i}))^2}{\\sum_{i=1}^{N}(log(e_{i})-log(\\bar{e})^2}-1)*-1

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: log Nash-Sutcliff model efficiency
    :rtype: float

    """
    if len(evaluation) == len(simulation):
        return float(1 - sum((np.log(simulation) - np.log(evaluation))**2) / sum((np.log(evaluation) - np.mean(np.log(evaluation)))**2))
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def log_p(evaluation=None, simulation=None, scale=0.1):
    """
    Logarithmic probability distribution

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Logarithmic probability distribution
    :rtype: float
    """
    #from scipy import stats
    #logLik = np.mean( stats.norm.logpdf(evaluation, loc=simulation, scale=.1) )
    scale = np.mean(evaluation) / 10
    if scale < .01:
        scale = .01
    if len(evaluation) == len(simulation):
        y = (np.array(evaluation) - np.array(simulation)) / scale
        normpdf = -y**2 / 2 - np.log(np.sqrt(2 * np.pi))
        return np.mean(normpdf)
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def correlationcoefficient(evaluation, simulation):
    """
    Correlation Coefficient

        .. math::

         r = \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}}

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Corelation Coefficient
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        Corelation_Coefficient = np.corrcoef(evaluation, simulation)[0, 1]
        return Corelation_Coefficient
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def rsquared(evaluation, simulation):
    """
    Coefficient of Determination

        .. math::

         r^2=(\\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})^2

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Coefficient of Determination
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        return correlationcoefficient(evaluation, simulation)**2
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def mse(evaluation, simulation):
    """
    Mean Squared Error

        .. math::

         MSE=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Mean Squared Error
    :rtype: float
    """

    if len(evaluation) == len(simulation):

        MSE_values = []

        for i in range(len(evaluation)):
            MSE_values.append((simulation[i] - evaluation[i])**2)

        MSE_sum = np.sum(MSE_values[0:len(evaluation)])

        MSE = MSE_sum / (len(evaluation))
        return MSE
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def rmse(evaluation, simulation):
    """
    Root Mean Squared Error

        .. math::

         RMSE=\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Root Mean Squared Error
    :rtype: float
    """
    if len(evaluation) == len(simulation) > 0:
        return np.sqrt(mse(evaluation, simulation))
    else:
        logging.warning("evaluation and simulation lists do not have the same length.")
        return np.nan


def mae(evaluation, simulation):
    """
    Mean Absolute Error

        .. math::

         MAE=\\frac{1}{N}\\sum_{i=1}^{N}(\\left |  e_{i}-s_{i} \\right |)

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Mean Absolute Error
    :rtype: float
    """
    if len(evaluation) == len(simulation) > 0:

        MAE_values = []

        for i in range(len(evaluation)):
            MAE_values.append(np.abs(simulation[i] - evaluation[i]))

        MAE_sum = np.sum(MAE_values[0:len(evaluation)])

        MAE = MAE_sum / (len(evaluation))

        return MAE
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def rrmse(evaluation, simulation):
    """
    Relative Root Mean Squared Error

        .. math::   

         RRMSE=\\frac{\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}}{\\bar{e}}

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Relative Root Mean Squared Error
    :rtype: float
    """

    if len(evaluation) == len(simulation):

        RRMSE = rmse(evaluation, simulation) / np.mean(evaluation)
        return RRMSE

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def agreementindex(evaluation, simulation):
    """
    Agreement Index (d) developed by Willmott (1981)

        .. math::   

         d = 1 - \\frac{\\sum_{i=1}^{N}(e_{i} - s_{i})^2}{\\sum_{i=1}^{N}(\\left | s_{i} - \\bar{e} \\right | + \\left | e_{i} - \\bar{e} \\right |)^2}  


    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Agreement Index
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        simulation, evaluation = np.array(simulation), np.array(evaluation)
        Agreement_index = 1 - (np.sum((evaluation - simulation)**2)) / (np.sum(
            (np.abs(simulation - np.mean(evaluation)) + np.abs(evaluation - np.mean(evaluation)))**2))
        return Agreement_index
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def covariance(evaluation, simulation):
    """
    Covariance

        .. math::
         Covariance = \\frac{1}{N} \\sum_{i=1}^{N}((e_{i} - \\bar{e}) * (s_{i} - \\bar{s}))

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Covariance
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        Covariance_values = []

        for i in range(len(evaluation)):
            Covariance_values.append(
                (evaluation[i] - np.mean(evaluation)) * (simulation[i] - np.mean(simulation)))

        Covariance = np.sum(Covariance_values) / (len(evaluation))
        return Covariance
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def decomposed_mse(evaluation, simulation):
    """
    Decomposed MSE developed by Kobayashi and Salam (2000)

        .. math ::
         dMSE = (\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i}))^2 + SDSD + LCS

         SDSD = (\\sigma(e) - \\sigma(s))^2

         LCS = 2 \\sigma(e) \\sigma(s) * (1 - \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Decomposed MSE
    :rtype: float
    """

    if len(evaluation) == len(simulation):

        Decomposed_MSE = str(round((bias(evaluation, simulation))**2, 2)) + '(bias**2) + ' + str(round((_standarddeviation(evaluation) - _standarddeviation(simulation))**2, 2)) + \
            '(SDSD) + ' + str(round(2 * _standarddeviation(evaluation) * _standarddeviation(
                simulation) * (1 - correlationcoefficient(evaluation, simulation)), 2)) + '(LCS)'

        return Decomposed_MSE
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def kge(evaluation, simulation, return_all=False):
    """
    Kling-Gupta Efficiency

    Corresponding paper: 
    Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling

    output:
        kge: Kling-Gupta Efficiency
    optional_output:
        cc: correlation 
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """
    if len(evaluation) == len(simulation):
        cc = np.corrcoef(evaluation, simulation)[0, 1]
        alpha = np.std(simulation) / np.std(evaluation)
        beta = np.sum(simulation) / np.sum(evaluation)
        kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        if return_all:
            return kge, cc, alpha, beta
        else:
            return kge
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def rsr(evaluation, simulation):
    """
    RMSE-observations standard deviation ratio 

    Corresponding paper: 
    Moriasi, Arnold, Van Liew, Bingner, Harmel, Veith, 2007, Model Evaluation Guidelines for Systematic Quantification of Accuracy in Watershed Simulations

    output:
        rsr: RMSE-observations standard deviation ratio 
    """
    if len(evaluation) == len(simulation):
        rmse_temp = rmse(evaluation, simulation)
        std = _standarddeviation(evaluation)
        rsr = rmse_temp / std
        return rsr
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def _variance(evaluation):
    """
    Variance

        .. math:: 

         Variance = \\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :return: Variance
    :rtype: float
    """
    Variance_values = []
    for i in range(len(evaluation)):
        Variance_values.append((evaluation[i] - np.mean(evaluation))**2)
    Variance = np.sum(Variance_values) / len(evaluation)
    return Variance


def _standarddeviation(evaluation):
    """
    Standard Derivation (sigma)

        .. math::
         sigma = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2}

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Standard Derivation
    :rtype: float
    """

    return np.sqrt(_variance(evaluation))


def volume_error(evaluation, simulation):
    """
    Returns the Volumer Error (Ve).
    It is an indicator of the agreement between the averages of the simulated
    and observed runoff (i.e. long-term water balance).
    used in this paper:
    Reynolds, J.E., S. Halldin, C.Y. Xu, J. Seibert, and A. Kauffeldt. 2017.
    “Sub-Daily Runoff Predictions Using Parameters Calibrated on the Basis of Data with a 
    Daily Temporal Resolution.” Journal of Hydrology 550 (July):399–411. 
    https://doi.org/10.1016/j.jhydrol.2017.05.012.

        .. math::

         Sum(simulation-evaluation)/sum(simulation)
    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Volume Error
    :rtype: float
    """
    if len(evaluation) == len(simulation):

        ve = np.sum(simulation - evaluation) / np.sum(evaluation)

        return float(ve)

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan

_all_functions = [agreementindex, bias, correlationcoefficient, covariance, decomposed_mse,
                  kge, log_p, lognashsutcliffe, mae, mse, nashsutcliffe, pbias, rmse, rrmse, rsquared,
                  rsr, volume_error
                  ]

def calculate_all_functions(evaluation, simulation):
    """
    Calculates all objective functions from spotpy.objectivefunctions
    and returns the results as a list of name/value pairs

    :param evaluation: a sequence of evaluation data
    :param simulation: a sequence of simulation data
    :return: A list of (name, value) tuples
    """

    result = []
    for f in _all_functions:
        # Check if the name is not private and attr is a function but not this

        try:
            result.append((f.__name__, f(evaluation, simulation)))
        except:
            result.append((f.__name__, np.nan))

    return result
