# -*- coding: utf-8 -*-
'''
Copyright 2017 by Tobias Houska, Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Benjamin Manns
This module contains a framework to summarize the distance between the model simulations and corresponding observations
by calculating likelihood values.
We modified the formula so, that a best fit of model can be archived by maximizing negative likelihood to zero
'''

#### TEST RESULTS WITH DREAM and hymod.py ####
# Dream has now a bunch of options. We tested only likelihoods, which do not need a additional parameter
# from this we saw that
# ExponentialTransformErrVarShapingFactor with option 2 and 6 is good
# InverseErrorVarianceShapingFactor with option 2 ist good
# gaussianLikelihoodMeasErrorOut with option 6 is good
# NoisyABCGaussianLikelihood with Option 6 is good
# ABCBoxcarLikelihood with Option 2 and 6 is good
# logLikelihood with option 2 and 6 is good

import numpy as np
import math
import warnings


class LikelihoodError(Exception):
    """
    Define an own error class to know it is an error made by a likelihood calculation to warn the use for wrong inputs
    """
    pass


def __generateMeaserror(data):
    return np.array(data) * 0.1


def __calcSimpleDeviation(data, comparedata):
    __standartChecksBeforeStart(data, comparedata)
    d = np.array(data)
    c = np.array(comparedata)
    return d - c


def __standartChecksBeforeStart(data, comparedata):
    # some standard checks
    if data.__len__() != comparedata.__len__():
        raise LikelihoodError("Simulation and observation data have not the same length")
    if data.__len__() == 0:
        raise LikelihoodError("Data with no content can not be used as a foundation of calculation a likelihood")


def __jitter_measerror_if_needed(fun_name, measerror):
    size = measerror[measerror == 0.0].size
    if size > 0:
        warnings.warn(
            "[" + fun_name + "] realized that there are distinct distributed values. "
                             "We jittered the values but the result can be far away from the truth.")

        measerror[measerror == 0.0] = np.random.uniform(0.01, 0.1, size)
    return measerror


class TimeSeries:
    """
    The formulae are based on 2002-Brockwell-Introduction Time Series and Forecasting.pdf, pages 17-18
    and is available on every standard statistic literature
    """

    @staticmethod
    def acf(data, lag):
        """
        For a detailed explanation and more background information, please look into "Zeitreihenanalyse", pages 17-18,
        by Matti Schneider, Sebastian Mentemeier, SS 2010

        .. math::

            acf(h) = \\frac{1}{n} \\sum_{t=1}^{n-h}(x_{t+h}-mean(x))(x_t-mean(x))

        :param data: numerical values whereof a acf at lag `h` should be calculated
        :type data: list
        :param lag: lag defines how many steps between each values should be taken to where of a of correlation should be calculated

        :type lag: int
        :return: auto covariation of the data at lag `lag`
        :rtype: float
        """

        len = data.__len__()
        if len <= 0:
            raise LikelihoodError("Data with no content can not be used to calc autokorrelation")
        if lag is None or type(lag) != type(1):
            raise LikelihoodError("The lag musst be an integer")
        if lag > len:
            raise LikelihoodError("The lag can not be bigger then the size of your data")
        m = np.mean(data)
        d = np.array(data)
        # R-Style numpy inline sum
        return np.sum((d[lag:len] - m) * (d[0:len - lag] - m)) / len

    @staticmethod
    def AR_1_Coeff(data):
        """
        The autocovariance coefficient called as rho, for an AR(1) model can be calculated as shown here:

        .. math::

            \\rho(1) = \\frac{\\gamma(1)}{\\gamma(0)}

        For further information look for example in "Zeitreihenanalyse", pages 17, by Matti Schneider, Sebastian Mentemeier,
        SS 2010.

        :param data: numerical list
        :type data: list
        :return: autocorrelation coefficient
        :rtype: float
        """
        return TimeSeries.acf(data, 1) / TimeSeries.acf(data, 0)


def logLikelihood(data, comparedata, measerror=None):
    """
    This formula is based on the gaussian likelihood: homo/heteroscedastic data error formula which can be used in both
    cases if the data has a homo- or heteroscedastic data error. To archive numerical stability a log-transformation was done, 
    which derives following formula, as shown in formular 8 in: Vrugt 2016 Markov chain Monte Carlo 
    simulation using the DREAM software package: Theory, concepts, and Matlab implementation, EMS:


    .. math::

            p = \\frac{n}{2}\\log(2\\cdot\\pi)+\\sum_{t=1}^n \\log(\\sigma_t)+0.5\\cdot\\sum_{t=1}^n (\\frac{y_t-y_t(x)}{\\sigma_t})^2


    `Usage:` Maximizing the likelihood value guides to the best model. To do so, we modified the original formula of the
    paper.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: measurement errors of every data input, if nothing is given a standart calculation is done to simulate measurement errors
    :type measerror: list
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)
    data = np.array(data)
    comparedata = np.array(comparedata)
    if measerror is None:
        measerror = __generateMeaserror(data)
    measerror = np.array(measerror)
    measerror = __jitter_measerror_if_needed("logLikelihood", measerror)

    # TODO: Maximize is done but in positive way (from negative to zero is hard)
    return -data.__len__() / 2 * np.log(2 * np.pi) - np.nansum(np.log(measerror)) - 0.5 * np.sum(
        ((data - comparedata) / measerror) ** 2)


def gaussianLikelihoodMeasErrorOut(data, comparedata):
    """
    This formular called `Gaussian likelihood: measurement error integrated out` and simply calculates


    .. math::

            p = -n/2\\log(\\sum_{t=1}^n e_t(x)^2)

    with :math:`e_t` is the error residual from `data` and `comparedata`


    `Usage:` Maximizing the likelihood value guides to the best model.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)
    errorArr = np.array(__calcSimpleDeviation(data, comparedata))

    return -data.__len__() / 2 * np.log(np.sum(errorArr ** 2))


def gaussianLikelihoodHomoHeteroDataError(data, comparedata, measerror=None):
    """
    Assuming the data error is normal distributed with zero mean and sigma is the measerror, the standart deviation of
    the meassurment errors
    This formulation allows for homoscedastic (constant variance) and heteroscedastic measuresment errors
    (variance dependent on magnitude of data).

    .. math::

            p = \\prod_{t=1}^{n}\\frac{1}{\\sqrt{2\\pi\\sigma_t^2}}exp(-0.5(\\frac{\\bar y_t - y_t(x) }{sigma_t})^2)


    `Usage:` Maximizing the likelihood value guides to the best model.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: measurement errors of every data input, if nothing is given a standart calculation is done to simulate measurement errors
    :type measerror: list
    :return: the p value as a likelihood
    :rtype: float
    """
    # With the assumption that the error residuals are uncorrelated
    __standartChecksBeforeStart(data, comparedata)
    n = data.__len__()
    data = np.array(data)
    comparedata = np.array(comparedata)
    if measerror is None:
        measerror = __generateMeaserror(data)
    measerror = np.array(measerror)
    measerror = __jitter_measerror_if_needed("gaussianLikelihoodHomoHeteroDataError", measerror)

    # TODO Maximizing with negative to zero?
    # original: -np.prod((1 / (np.sqrt(2 * np.pi * measerror**2)))*np.exp(-0.5 * ((data-comparedata)/(measerror))**2))
    return -np.sum(
        (1 / (np.sqrt(2 * np.pi * measerror ** 2))) * np.exp(-0.5 * ((data - comparedata) / (measerror)) ** 2))


def LikelihoodAR1WithC(data, comparedata, measerror=None, params=None):
    """

    Suppose the error residuals assume an AR(1)-process


    .. math::

            e_t(x)=c+\\phi e_{t-1}(x)+\\eta_t

    with :math:`\\eta_t \\sim N(0,\sigma^2)`, and expectation :math:`E(e_t(x))=c/(1-\\phi)` and variance :math:`\\sigma^2/(1-\\phi^2)`


    This lead to the following standard `log-likelihood`:


    .. math::

            p = -n/2*\\log(2\\pi)-0.5*\\log(\\sigma_1^2/(1-\\phi^2))-\\frac{(e_1(x)-(c/(1-\\phi)))^2}{2\\sigma^2/(1-\\phi^2)}-\\sum_{t=2}^{n}\\log(\\sigma_t)-0.5\\sum_{t=2}^{n}(\\frac{(e_t(x)-c-\\phi e_{t-1}(x))}{\\sigma_t})^2

    `Usage:` Maximizing the likelihood value guides to the best model.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: measurement errors of every data input, if nothing is given a standart calculation is done to simulate measurement errors
    :type measerror: list
    :param params: Contains a tuple of model parameter which are needed for calculating the likelihood. Where the first component contains the values and the second the names of the valules.
        Following parameter are needed in this function:

        -1 < `likelihood_phi` < 1
    :type params: tuple
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)
    n = data.__len__()
    if measerror is None:
        measerror = __generateMeaserror(data)

    measerror = np.array(measerror)
    measerror = __jitter_measerror_if_needed("LikelihoodAR1WithC", measerror)

    paramDependencies = ["likelihood_phi"]

    if params is None:
        phi = TimeSeries.AR_1_Coeff(data)
    else:
        missingparams = []
        randomparset, parameternames = params
        randomparset = np.array(randomparset)
        parameternames = np.array(parameternames)
        for nm in paramDependencies:
            if nm not in parameternames:
                missingparams.append(nm)

        if missingparams.__len__() > 0:
            raise LikelihoodError(
                "Unfortunately contains your param list not all parameters which are needed for this class."
                "Following parameter are needed, too: " + str(missingparams))

        phi = float(randomparset[parameternames == 'likelihood_phi'])
    # Break the calculation if given parameter are not valid
    if abs(phi) >= 1:
        warnings.warn("The parameter 'phi' should be real between -1 and 1 and is: " + str(phi))
        return np.NAN

    expect = np.nanmean(data)
    errorArr = np.array(__calcSimpleDeviation(data, comparedata))
    c = expect * (1 - phi)

    # I summarize from 2 to n, but range starts in 1 (in python it is zero index), so just shift it with one
    n = data.__len__()
    sum_1 = np.sum(np.log(measerror[1:]))

    sum_2 = np.sum(((errorArr[1:] - c - phi * errorArr[:-1]) / (measerror[1:])) ** 2)

    # TODO Its maximaized but maybe from negative to zero, is that possible?
    return -(-(n / 2) * np.log(2 * np.pi) - 0.5 * np.log(measerror[0] ** 2 / (1 - phi ** 2)) - (
            (errorArr[0] - (c / (1 - phi))) ** 2 / (2 * measerror[0] ** 2 / (1 - phi ** 2))) - sum_1 - 0.5 * sum_2)


def LikelihoodAR1NoC(data, comparedata, measerror=None, params=None):
    """

    Based on the formula in `LikelihoodAR1WithC` we assuming that :math:`c = 0` and that means that the formula of `log-likelihood` is:

    .. math::

            p = -n/2*\\log(2\\pi)+0.5\\log(1-\\phi^2)-0.5(1-\\phi^2)\\sigma_1^{-2}e_1(x)^2-\\sum_{t=2}^{n}\\log(\\sigma_t)-0.5\\sum_{t=2}^{n}(\\frac{e_t(x)-\\phi e_{t-1}(x)}{\\sigma_t})^2

    `Usage:` Maximizing the likelihood value guides to the best model.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: measurement errors of every data input, if nothing is given a standart calculation is done to simulate measurement errors
    :type measerror: list
    :param params: Contains a tuple of model parameter which are needed for calculating the likelihood. Where the first component contains the values and the second the names of the valules.
        Following parameter are needed in this function:

        -1 < `likelihood_phi` < 1
    :type params: tuple
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)
    n = data.__len__()
    if measerror is None:
        measerror = __generateMeaserror(data)

    errorArr = np.array(__calcSimpleDeviation(data, comparedata))

    # I summarize from 2 to n, but range starts in 1 (in python it is zero index), so just shift it with one
    measerror = np.array(measerror)
    measerror = __jitter_measerror_if_needed("LikelihoodAR1NoC", measerror)

    paramDependencies = ["likelihood_phi"]

    if params is None:
        phi = np.random.uniform(-0.99, 0.99, 1)
    else:
        missingparams = []
        randomparset, parameternames = params
        for nm in paramDependencies:
            if nm not in parameternames:
                missingparams.append(nm)
        if missingparams.__len__() > 0:
            raise LikelihoodError(
                "Unfortunately contains your param list not all parameters which are needed for this class."
                "Following parameter are needed, too: " + str(missingparams))

        parameternames = np.array(parameternames)
        randomparset = np.array(randomparset)
        phi = float(randomparset[parameternames == 'likelihood_phi'])

        # Break the calculation if given parameter are not valid
        if abs(phi) >= 1:
            warnings.warn("The parameter 'phi' should be real between -1 and 1 and is: " + str(phi))
            return np.NAN

    sum_1 = np.sum(np.log(measerror[1:]))
    sum_2 = np.sum(((errorArr[1:] - phi * errorArr[:-1]) / measerror[1:]) ** 2)

    # TODO Maximizing with negative to zero?
    return -float(
        -(n / 2) * np.log(2 * np.pi) + 0.5 * np.log(1 - phi ** 2) - 0.5 * (1 - phi ** 2) * (1 / measerror[0] ** 2) * \
        errorArr[0] ** 2 - sum_1 - 0.5 * sum_2)


def generalizedLikelihoodFunction(data, comparedata, measerror=None, params=None):
    """
    Under the assumption of having correlated, heteroscedastic, and non‐Gaussian errors and assuming that the data are
    coming from a time series modeled as

    .. math::

            \\Phi_p(B)e_t = \\sigma_t a_t

    with `a_t` is an i.i.d. random error with zero mean and unit standard deviation, described by a skew exponential
    power (SEP) density the likelihood `p` can be calculated as follows:


    .. math::

            p = \\frac{2\\sigma_i}{\\xi+\\xi^{-1}}\\omega_\\beta exp(-c_\\beta |a_{\\xi,t}|^{2/(1+\\beta)})


    where

     .. math::

            a_{\\xi,t} = \\xi^{-sign(\\mu_\\xi+\\sigma_\\xi a_t )}(\\mu_\\xi+\\sigma_\\xi a_t)


    For more detailes see: http://onlinelibrary.wiley.com/doi/10.1029/2009WR008933/epdf, page 3, formualar (6) and pages 15, Appendix A.

    `Usage:` Maximizing the likelihood value guides to the best model.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: measurement errors of every data input, if nothing is given a standart calculation is done to
        simulate measurement errors
    :type measerror: list
    :param params: Contains a tuple of model parameter which are needed for calculating the likelihood. Where the first component contains the values and the second the names of the valules.
        Following parameter are needed in this function:

        -1 < `likelihood_beta`   < 1,

        0  < `likelihood_xi`    <= 10,

        0 <= `likelihood_sigma0` <= 1,

        0 <= `likelihood_sigma1` <= 1,

        0 <= `likelihood_phi1`    < 1,

        0 <= `likelihood_muh`    <= 100
    :type params: tuple
    :return: the p value as a likelihood
    :rtype: float
    """

    __standartChecksBeforeStart(data, comparedata)
    errorArr = __calcSimpleDeviation(data, comparedata)
    if measerror is None:
        measerror = __generateMeaserror(data)
    measerror = np.array(measerror)
    comparedata = np.array(comparedata)
    measerror = __jitter_measerror_if_needed("generalizedLikelihoodFunction", measerror)

    paramDependencies = ["likelihood_beta", "likelihood_xi", "likelihood_sigma0", "likelihood_sigma1",
                         "likelihood_phi1", "likelihood_muh"]

    if params is None:
        # for this params look into http://onlinelibrary.wiley.com/doi/10.1029/2009WR008933/epdf, page 5
        beta = np.random.uniform(-0.99, 1, 1)
        xi = np.random.uniform(0.01, 10, 1)
        sigma0 = np.random.uniform(0, 1, 1)
        sigma1 = np.random.uniform(0, 1, 1)
        phi1 = np.random.uniform(0, .99, 1)
        muh = np.random.uniform(0, 100, 1)

    else:
        missingparams = []
        randomparset, parameternames = params
        parameternames = np.array(parameternames)
        randomparset = np.array(randomparset)

        for nm in paramDependencies:
            if nm not in parameternames:
                missingparams.append(nm)

        if missingparams.__len__() > 0:
            raise LikelihoodError(
                "Unfortunately contains your param list not all parameters which are needed for this class."
                "Following parameter are needed, too: " + str(missingparams))

        beta = float(randomparset[np.where(parameternames == 'likelihood_beta')])
        xi = float(randomparset[np.where(parameternames == 'likelihood_xi')])
        sigma0 = float(randomparset[np.where(parameternames == 'likelihood_sigma0')])
        sigma1 = float(randomparset[parameternames == 'likelihood_sigma0'])
        phi1 = float(randomparset[np.where(parameternames == 'likelihood_phi1')])
        muh = float(randomparset[np.where(parameternames == 'likelihood_muh')])

        # Break the calculation if given parameter are not valid
        if beta <= -1 or beta > 1:
            warnings.warn("The parameter 'beta' should be greater then -1 and less equal 1 and is: " + str(beta))
            return np.NAN
        if xi < 0.1 or xi > 10:
            warnings.warn("The parameter 'xi' should be between 0.1 and 10 and is: " + str(xi))
            return np.NAN
        if sigma0 < 0 or sigma0 > 1:
            warnings.warn("The parameter 'sigma0' should be between 0 and 1 and is: " + str(sigma0))
            return np.NAN
        if sigma1 < 0 or sigma1 > 1:
            warnings.warn("The parameter 'sigma1' should be between 0 and 1 and is: " + str(sigma1))
            return np.NAN
        if phi1 < 0 or phi1 > 1:
            warnings.warn("The parameter 'phi1' should be between 0 and 1 and is: " + str(phi1))
            return np.NAN
        if muh < 0 or muh > 100:
            warnings.warn("The parameter 'muh' should be between 0 and 100 and is: " + str(muh))
            return np.NAN

    try:
        omegaBeta = np.sqrt(math.gamma(3 * (1 + beta) / 2)) / ((1 + beta) * np.sqrt(math.gamma((1 + beta) / 2) ** 3))
        M_1 = math.gamma(1 + beta) / (np.sqrt(math.gamma(3 * (1 + beta) / 2)) * np.sqrt(math.gamma((1 + beta) / 2)))
        M_2 = 1
        sigma_xi = np.sqrt(np.abs(float((M_2 - M_1 ** 2) * (xi ** 2 + xi ** (-2)) + 2 * M_1 ** 2 - M_2)))
        cBeta = (math.gamma(3 * (1 + beta) / 2) / math.gamma((1 + beta) / 2)) ** (1 / (1 + beta))
    except ValueError:
        raise LikelihoodError("Please check your parameter input there is something wrong with the parameter")

    if xi != 0.0:
        mu_xi = M_1 * (xi - (xi ** (-1)))
    else:
        mu_xi = 0.0

    n = data.__len__()

    sum_at = 0
    # formula for a_t is from page 3, (6)
    for j in range(n - 1):
        t = j + 1
        if t > 0 and t < n and type(t) == type(1):
            a_t = (errorArr[t] - phi1 * errorArr[t - 1]) / (measerror[t])
        else:
            warnings.warn("Your parameter 't' does not suit to the given data list")
            return None

        a_xi_t = xi ** (-1 * np.sign(mu_xi + sigma_xi * a_t)) * (mu_xi + sigma_xi * a_t)

        sum_at += np.abs(a_xi_t) ** (2 / (1 + beta))

    # page 3 formula 5 of this paper explain that sigma[t] = sigma0 + sigma1*E[t]
    # where E[t] is called y(x) in the main paper (discrepancy) and sigma0 and sigma1 are input parameter which also
    # can be generate by the function itself. Then
    # E[t] = Y_{ht}*mu[t]
    # where Y_{ht} should be the simulated model data and mu_t = exp(mu_h * Y_{ht}).
    # So, mu_h is "a bias parameter to be inferred from the model." (cite, page 3, formula (3))

    mu_t = np.exp(muh * comparedata)

    E = comparedata * mu_t

    sigmas = sigma0 + sigma1 * E
    if sigmas[sigmas <= 0.0].size > 0:
        warnings.warn("Sorry, you comparedata have negative values. Maybe you model has some inaccurate"
                      " assumptions or there is another error."
                      " We cannot calculate this likelihood")
        return np.NAN

    return n * np.log(omegaBeta * (2 * sigma_xi) / np.abs(xi + (1 / xi))) - np.sum(np.log(sigmas)) - cBeta * sum_at


def LaplacianLikelihood(data, comparedata, measerror=None):
    """
    This likelihood function is based on
    https://www.scopus.com/record/display.uri?eid=2-s2.0-0000834243&origin=inward&txGid=cb49b4f37f76ce197f3875d9ea216884
    and use this formula

    .. math::

            p = -\\sum_{t=1}^n \\log(2\\sigma_t)-\\sum_{t=1}^n (\\frac{|e_t(x)|}{\\sigma_t})

    `Usage:` Maximizing the likelihood value guides to the best model,
    because the less :math:`\\sum_{t=1}^n (\\frac{|e_t(x)|}{\\sigma_t})`
    is the better fits the model simulation data to the observed data.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: measurement errors of every data input, if nothing is given a standart calculation is done to
        simulate measurement errors
    :type measerror: list
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)
    errArr = np.array(__calcSimpleDeviation(data, comparedata))
    if measerror is None:
        measerror = __generateMeaserror(data)
    measerror = np.array(measerror)
    measerror = __jitter_measerror_if_needed("LaplacianLikelihood", measerror)

    # Log from negative value makes no sense at all
    return -1 * np.sum(np.log(2 * np.abs(measerror))) - np.sum(np.abs(errArr) / measerror)


def SkewedStudentLikelihoodHomoscedastic(data, comparedata, measerror=None):
    """
    Under the assumption that the data are homoscedastic, i.e. the they have a constant measurement error and that the
    residuals :math:`\\epsilon_i` follow a Gaussian distribution we can determine the likelihood by calculation this:

     .. math::

            p = \\prod_{i=1}^n \\frac{1}{\\sqrt{2\\pi}\\sigma_{const}}exp(-\\frac{\\epsilon_i}{2})

    For detailed mathematical question take a look into hessd-12-2155-2015.pdf
    (https://www.hydrol-earth-syst-sci-discuss.net/12/2155/2015/hessd-12-2155-2015.pdf) pages 2164-2165

    `Usage:` Maximizing the likelihood value guides to the best model. Be aware that only a right model
    assumption leads to a result which makes sense.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: a constant measurement error
    :type measerror: int
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)
    if measerror is None:
        measerror = __generateMeaserror(data)

    measerror = np.mean(measerror)

    res = np.array(__calcSimpleDeviation(data, comparedata))

    # TODO Maximizing with negative to zero?
    # Original: -np.prod(1 / (np.sqrt(2 * np.pi) * measerror) * np.exp(-1 * (res ** 2) / (2)))
    return -np.sum((1 / (np.sqrt(2 * np.pi) * measerror) * np.exp(-1 * (res ** 2) / (2))))


def SkewedStudentLikelihoodHeteroscedastic(data, comparedata, measerror=None, params=None):
    """
    Under the assumption that the data are heteroscedastic, i.e. the they have for every measurement another error and
    that the residuals are non-Gaussian distributed we perform a likelihoodcalculation based on this formualar, having
    :math:`k` as the skewness parameter from the data and where we assume that the kurtosis parameter :math:`\\nu > 2`:


     .. math::

            p = \\prod_{i=1}^n p_i


    Where

    .. math::

            \\eta_i = (\\epsilon_i-\\epsilon_{i-1}\\phi)\\sqrt{1-\\phi^2}

    and

    .. math::

            p_i = \\frac{2c_2\\Gamma(\\frac{\\nu+1}{2})\\sqrt{\\frac{\\nu}{\\nu-2}}}{\\Gamma(\\frac{\\nu}{2})\\sqrt{\\pi \\nu}\\sqrt{1-\\phi^2}\\sigma_i} \\times (1+\\frac{1}{\\nu-2}(\\frac{c_1+c_2+eta_i}{k^{sign(c_1+c_2+eta_i)}})^2)^{-\\frac{\\nu+1}{2}}


    and

    .. math::

            c_1 = \\frac{(k^2-\\frac{1}{2})2\\Gamma(\\frac{\\nu+1}{2})\\sqrt{\\frac{\\nu}{\\nu-2}}(\\nu-2)}{k+\\frac{1}{k}\\Gamma(\\frac{\\nu}{2})\\sqrt{\\pi \\nu}(\\nu-1)}


    and

    .. math::

            c_2 = \\sqrt{-c_1^2+\\frac{k^3+\\frac{1}{k^3}}{k+\\frac{1}{k}}}


    For detailed mathematical question take a look into hessd-12-2155-2015.pdf
    (https://www.hydrol-earth-syst-sci-discuss.net/12/2155/2015/hessd-12-2155-2015.pdf) pages 2165-2169, formular (15).

    `Usage:` Maximizing the likelihood value guides to the best model. Be aware that only a right model asumption leads to
    a result which makes sense.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: measurement errors of every data input, if nothing is given a standart calculation is done to simulate measurement errors
    :type measerror: list
    :param params: Contains a tuple of model parameter which are needed for calculating the likelihood. Where the first component contains the values and the second the names of the valules.
        Following parameter are needed in this function:

        `likelihood_nu` > 2

        `likelihood_kappa` > 0

        -1 < `likelihood_phi` < 1
    :type params: tuple
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)
    if measerror is None:
        measerror = __generateMeaserror(data)

    measerror = np.array(measerror)
    measerror = __jitter_measerror_if_needed("SkewedStudentLikelihoodHeteroscedastic", measerror)

    diff = np.array(__calcSimpleDeviation(data, comparedata))

    paramDependencies = ["likelihood_nu", "likelihood_kappa", "likelihood_phi"]

    if params is None:
        # based on VRUGTS paper, footnote "YING", page 307
        nu = np.random.uniform(2.001, 100, 1)
        k = np.random.uniform(0.001, 100, 1)
        phi = np.random.uniform(-0.99, 0.99, 1)

    else:
        missingparams = []
        randomparset, parameternames = params

        randomparset = np.array(randomparset)
        parameternames = np.array(parameternames)

        for nm in paramDependencies:
            if nm not in parameternames:
                missingparams.append(nm)

        if missingparams.__len__() > 0:
            raise LikelihoodError(
                "Unfortunately contains your param list not all parameters which are needed for this class."
                "Following parameter are needed, too: " + str(missingparams))

        nu = randomparset[parameternames == 'likelihood_nu'][0]
        k = randomparset[parameternames == 'likelihood_kappa'][0]
        phi = randomparset[parameternames == 'likelihood_phi'][0]

    if abs(phi) > 1:
        warnings.warn(
            "[SkewedStudentLikelihoodHeteroscedastic] The parameter 'phi' should be between -1 and 1 and is: " + str(
                phi))
        return np.NAN
    if nu <= 2:
        warnings.warn(
            "[SkewedStudentLikelihoodHeteroscedastic] The parameter 'nu' should be greater then 2 and is: " + str(
                nu))
        return np.NAN
    if k <= 0:
        warnings.warn(
            "[SkewedStudentLikelihoodHeteroscedastic] The parameter 'k' should be greater then 0 and is: " + str(
                k))
        return np.NAN

    eta_all = diff[1:] - phi * diff[:-1] * np.sqrt(1 - phi ** 2)
    c_1 = ((k ** 2 - 1 / (k ** 2)) * 2 * math.gamma((nu + 1) / 2) * np.sqrt(nu / (nu - 2)) * (nu - 2)) / (
            (k + (1 / k)) * math.gamma(nu / 2) * np.sqrt(np.pi * nu) * (nu - 1))

    for_c2 = -1 * (c_1) ** 2 + (k ** 3 + 1 / k ** 3) / (k + 1 / k)

    c_2 = np.sqrt(for_c2)

    # TODO Maximizing with negative to zero?
    return np.log(-np.prod((2 * c_2 * math.gamma((nu + 1) / 2) * np.sqrt(nu / (nu - 2))) / (
            (k + 1 / k) * math.gamma(nu / 2) * np.sqrt(np.pi * nu) * np.sqrt(1 - phi ** 2) * measerror[1:]) \
                           * (1 + (1 / (nu - 2)) * (
            (c_1 + c_2 * eta_all) / (k ** (np.sign(c_1 + c_2 * eta_all)))) ** 2) ** (
                                   -(nu + 1) / 2)))


def SkewedStudentLikelihoodHeteroscedasticAdvancedARModel(data, comparedata, measerror=None, params=None):
    """

    This function is based of the previos one, called `SkewedStudentLikelihoodHeteroscedastic`. We expand
    the AR(1) Model so that the expectation of :math:`\\eta_i` is equal to the expectation of a residual :math:`\\epsilon_i`.
    So we having

    .. math::

            \\eta_i = (\\epsilon_i-\\epsilon_{i-1}\\phi + \\frac{\\phi}{N}\\sum_{j = 1}^{N} \\epsilon_j)\\sqrt{1-\\phi^2}

    For detailed mathematical question take a look into hessd-12-2155-2015.pdf
    (https://www.hydrol-earth-syst-sci-discuss.net/12/2155/2015/hessd-12-2155-2015.pdf) pages 2170 formular (20).

    `Usage:` Maximizing the likelihood value guides to the best model. Be aware that only a right model asumption leads to
    a result which makes sense.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: measurement errors of every data input, if nothing is given a standart calculation is done to simulate measurement errors
    :type measerror: list
    :param params: Contains a tuple of model parameter which are needed for calculating the likelihood. Where the first component contains the values and the second the names of the valules.
        Following parameter are needed in this function:

        `likelihood_nu` > 2

        `likelihood_kappa` > 0

        -1 < `likelihood_phi` < 1
    :type params: tuple
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)
    if measerror is None:
        measerror = __generateMeaserror(data)

    measerror = np.array(measerror)
    measerror = __jitter_measerror_if_needed("SkewedStudentLikelihoodHeteroscedasticAdvancedARModel", measerror)

    res = np.array(__calcSimpleDeviation(data, comparedata))

    paramDependencies = ["likelihood_nu", "likelihood_kappa", "likelihood_phi"]
    if params is None:
        # based on VRUGTS paper, footnote "YING", page 307
        nu = np.random.uniform(2.001, 100, 1)
        k = np.random.uniform(0.001, 100, 1)
        phi = np.random.uniform(-0.99, 0.99, 1)

    else:
        missingparams = []
        randomparset, parameternames = params
        randomparset = np.array(randomparset)
        parameternames = np.array(parameternames)

        for nm in paramDependencies:
            if nm not in parameternames:
                missingparams.append(nm)

        if missingparams.__len__() > 0:
            raise LikelihoodError(
                "Unfortunately contains your param list not all parameters which are needed for this class."
                "Following parameter are needed, too: " + str(missingparams))

        nu = randomparset[parameternames == 'likelihood_nu'][0]
        k = randomparset[parameternames == 'likelihood_kappa'][0]
        phi = randomparset[parameternames == 'likelihood_phi'][0]

        if abs(phi) > 1:
            warnings.warn(
                "[SkewedStudentLikelihoodHeteroscedasticAdvancedARModel] The parameter 'phi' should be between -1 and 1 and is: " + str(
                    phi))
            return np.NAN
        if nu <= 2:
            warnings.warn(
                "[SkewedStudentLikelihoodHeteroscedasticAdvancedARModel] The parameter 'nu' should be greater then 2 and is: " + str(
                    nu))
            return np.NAN
        if k <= 0:
            warnings.warn(
                "[SkewedStudentLikelihoodHeteroscedasticAdvancedARModel] The parameter 'k' should be greater then 0 and is: " + str(
                    k))
            return np.NAN

    N = data.__len__()
    eta_all = (res[1:] - phi * res[:-1] + phi / (N) * np.sum(res)) * np.sqrt(1 - phi ** 2)

    c_1 = ((k ** 2 - 1 / (k ** 2)) * 2 * math.gamma((nu + 1) / 2) * np.sqrt(nu / (nu - 2)) * (nu - 2)) / (
                (k + (1 / k)) * math.gamma(nu / 2) * np.sqrt(np.pi * nu) * (nu - 1))
    for_c2 = -1 * (c_1) ** 2 + (k ** 3 + 1 / k ** 3) / (k + 1 / k)

    c_2 = np.sqrt(for_c2)

    # TODO Maximizing with negative to zero?
    datas = ((2 * c_2 * math.gamma((nu + 1) / 2) * np.sqrt(nu / (nu - 2))) / (
            (k + 1 / k) * math.gamma(nu / 2) * np.sqrt(np.pi * nu) * np.sqrt(1 - phi ** 2) * measerror[1:]) \
             * (1 + (1 / (nu - 2)) * (
                    (c_1 + c_2 * eta_all) / (k ** (np.sign(c_1 + c_2 * eta_all)))) ** 2) ** (
                     -(nu + 1) / 2))

    return np.log(-np.prod((2 * c_2 * math.gamma((nu + 1) / 2) * np.sqrt(nu / (nu - 2))) / (
            (k + 1 / k) * math.gamma(nu / 2) * np.sqrt(np.pi * nu) * np.sqrt(1 - phi ** 2) * measerror[1:]) \
                           * (1 + (1 / (nu - 2)) * (
            (c_1 + c_2 * eta_all) / (k ** (np.sign(c_1 + c_2 * eta_all)))) ** 2) ** (
                                   -(nu + 1) / 2)))


def NoisyABCGaussianLikelihood(data, comparedata, measerror=None):
    """
    The likelihood function is based on the Wald distribution, whose likelihood function is given by

    .. math::

            p = \\prod_{i=1}^N f(y_i|\\alpha,\\nu).


    A epsilon is used to define :math:`P(\\theta|\\rho(S_1(Y),S_2(Y(X))) < \\epsilon).
    Using the means of the standart observation is a good value for \\epsilon.

    An Euclidean distance calculation is used, which is based on https://www.reading.ac.uk/web/files/maths/Preprint_MPS_15_09_Prangle.pdf
    , page 2.

    `Usage:` Maximizing the likelihood value guides to the best model.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: measurement errors of every data input, if nothing is given a standart calculation is done to simulate measurement errors
    :type measerror: list
    :return: the p value as a likelihood
    :rtype: float
    """

    __standartChecksBeforeStart(data, comparedata)
    if measerror is None:
        measerror = __generateMeaserror(data)
    sigmas = np.array(measerror)
    measerror = np.mean(measerror)

    size = sigmas[sigmas == 0.0].size
    if size > 0:
        warnings.warn(
            "[NoisyABCGaussianLikelihood] reaslized that there are distinct distributed values. We jittered the values but the result can be far away from the truth.")
        sigmas[sigmas == 0.0] = np.random.uniform(0.01, 0.1, size)

    if measerror == 0.0:
        warnings.warn(
            "[NoisyABCGaussianLikelihood] reaslized that the mean of the measerror is zero and therefore is no likelihood calculation possible")
        return np.NAN

    m = data.__len__()
    data = np.array(data)
    comparedata = np.array(comparedata)

    # The euclidean distance has a bit diffrent formula then the original paper showed
    return -m / 2 * np.log(2 * np.pi) - m * np.log(measerror) - 0.5 * 1 / (measerror ** 2) * np.sqrt(
        np.sum(((data - comparedata) / sigmas) ** 2))


def ABCBoxcarLikelihood(data, comparedata, measerror=None):
    """
    A simple ABC likelihood function is the Boxcar likelihood given by the formular:
    
    .. math::

            p = \\max_{i=1}^N(\\epsilon_j - \\rho(S(Y),S(Y(X)))).
            
    :math:`\\rho(S(Y),S(Y(X)))` is the eucledean distance.

    `Usage:` Maximizing the likelihood value guides to the best model.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: measurement errors of every data input, if nothing is given a standart calculation is done to simulate measurement errors
    :type measerror: list
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)
    if measerror is None:
        measerror = __generateMeaserror(data)

    data = np.array(data)
    comparedata = np.array(comparedata)

    measerror = np.array(measerror)
    measerror = __jitter_measerror_if_needed("ABCBoxcarLikelihood", measerror)

    # Usage of euclidean distance changes the formula a bit

    # TODO Maximizing with negative to zero?
    return np.min(measerror - np.sqrt(((data - comparedata) / measerror) ** 2))


def LimitsOfAcceptability(data, comparedata, measerror=None):
    """
   This calculation use the generalized likelihood uncertainty estimation by counting all Euclidean distances which are
   smaller then the deviation of the measurement value.

    .. math::

            p=\\sum_{j=1}^m I(|\\rho(S_j(Y)-S_j(Y(X))| \\leq \\epsilon_j)

    Variable :math:`I(a)` returns one if `a` is true, zero otherwise.

    `Usage:` Maximizing the likelihood value guides to the best model.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param measerror: measurement errors of every data input, if nothing is given a standart calculation is done to simulate measurement errors
    :type measerror: list
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)
    if measerror is None:
        measerror = __generateMeaserror(data)

    data = np.array(data)
    comparedata = np.array(comparedata)

    measerror = np.array(measerror)
    measerror = __jitter_measerror_if_needed("LimitsOfAcceptability", measerror)

    # Use simple non euclidean but weighted distance measurement.
    return np.sum(np.abs((data - comparedata) / measerror) <= measerror)


def InverseErrorVarianceShapingFactor(data, comparedata, G=10):
    """
    This function simply use the variance in the error values (:math:`E(X)=Y-Y(X)`) as a likelihood value as this formula
    shows:

    .. math::

            p=-G \\log(Var(E(x)))

    The factor `G` comes from the DREAMPar model. So this factor can be changed according to the used model.

    For more details see also: http://onlinelibrary.wiley.com/doi/10.1002/hyp.3360060305/epdf.

    `Usage:` Maximize the likelihood value guides to the best model.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param G: DREAMPar model parameter `G`
    :type G: float
    :return: the p value as a likelihood
    :rtype: float
    """

    __standartChecksBeforeStart(data, comparedata)

    errArr = np.nanvar(np.array(__calcSimpleDeviation(data, comparedata)))
    if errArr == 0.0:
        warnings.warn(
            "[InverseErrorVarianceShapingFactor] reaslized that the variance in y(x)-y is zero and that makes no sence and also impossible to calculate the likelihood.")
        return np.NAN
    else:
        # Gives an better convergence, so close values are more less and apart values are more great.
        # (0 is the best so to say).
        return -G * np.log(errArr) ** 3


def NashSutcliffeEfficiencyShapingFactor(data, comparedata, G=10):
    """
    This function use the opposite ratio of variance of the error terms between observed and simulated and the variance
    of the observed data as a base to claculate the
    likelihood and transform the values with the logarithm.

    .. math::

            p=G\\cdot\\log(1-\\frac{Var(E(x)}{Var(Y)})

    The factor `G` comes from the DREAMPar model. So this factor can be changed according to the used model.

    For more details see also: http://onlinelibrary.wiley.com/doi/10.1029/95WR03723/epdf.

    `Usage:` Maximize the likelihood value guides to the best model. If the function return NAN, than you can not use this
    calculation method or the `comparedata` is too far away from `data`.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param G: DREAMPar model parameter `G`
    :type G: float
    :return: the p value as a likelihood
    :rtype: float
    """

    __standartChecksBeforeStart(data, comparedata)

    errArr = np.array(__calcSimpleDeviation(data, comparedata))

    if np.nanvar(data) == 0.0:
        warnings.warn(
            "[NashSutcliffeEfficiencyShapingFactor] reaslized that the variance of the data is zero. Thereforee is no likelihood calculation possible")
        return np.NAN
    else:
        ratio = np.nanvar(errArr) / np.nanvar(data)

        if ratio > 1:
            warnings.warn(
                "[NashSutcliffeEfficiencyShapingFactor]: The ratio between residual variation and observation "
                "variation is bigger then one and therefore"
                "we can not calculate the liklihood. Please use another function which fits to this data and / or "
                "model")
            return np.NAN
        else:
            return G * np.log(1 - ratio)


def ExponentialTransformErrVarShapingFactor(data, comparedata, G=10):
    """
    This function use the variance of the error terms between observed and simulated data as a base to claculate the
    likelihood.

    .. math::

            p=-G\\cdot Var(E(x))

    The factor `G` comes from the DREAMPar model. So this factor can be changed according to the used model.

    For more details see also: http://onlinelibrary.wiley.com/doi/10.1029/95WR03723/epdf.

    `Usage:` Maximize the likelihood value guides to the best model.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :param G: DREAMPar model parameter `G`
    :type G: float
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)

    errArr = np.array(__calcSimpleDeviation(data, comparedata))

    return -G * np.nanvar(errArr)


def sumOfAbsoluteErrorResiduals(data, comparedata):
    """
    This function simply calc the deviation between observed and simulated value and perform a log transform. Detailed
    information can be found in http://onlinelibrary.wiley.com/doi/10.1002/hyp.3360060305/epdf.

    .. math::

            p=-\\log(\\sum_{t=1}^n |e_t(x)|)

    `Usage:` Maximize the likelihood value guides to the best model.

    :param data: observed measurements as a numerical list
    :type data: list
    :param comparedata: simulated data from a model which should fit the original data somehow
    :type comparedata: list
    :return: the p value as a likelihood
    :rtype: float
    """
    __standartChecksBeforeStart(data, comparedata)

    errArr = np.array(__calcSimpleDeviation(data, comparedata))
    return -1 * np.log(np.sum(np.abs(errArr)))
