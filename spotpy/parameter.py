# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Philipp Kraft and Tobias Houska
Contains classes to generate random parameter sets
'''

import numpy.random as rnd
import numpy as np


class Base(object):
    """
    This is a universal random parameter class
    It creates a random number (or array) drawn from specified distribution
    """

    def __init__(self, name, rndfunc, rndargs, step=None, optguess=None, minbound=None, maxbound=None, *args, **kwargs):
        """
        :name:     Name of the parameter
        :rndfunc:  Function to draw a random number, 
                eg. the random functions from numpy.random
        :rndargs:  Argument-tuple for the random function
                eg. lower and higher bound 
                (number and meaning of arguments depends on the function)
                tuple is unpacked as args to rndfunc call
        :step:     (optional) number for step size required for some algorithms 
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of 
                rndfunc(*rndargs, size=1000) 
        """
        self.name = name
        self.rndfunc = rndfunc
        self.rndargs = rndargs
        self.step = step or np.percentile(self(size=1000), 50)
        self.optguess = optguess or (np.percentile(self(size=1000), 50) -
                                     np.percentile(self(size=1000), 40))
        self.minbound = minbound or np.min(self(size=1000))
        self.maxbound = maxbound or np.max(self(size=1000))

    def __call__(self, **kwargs):
        """
        Returns a parameter realization
        """
        return self.rndfunc(*self.rndargs, **kwargs)

    def astuple(self):
        """
        Returns a tuple of a realization and the other parameter properties
        """
        return self(), self.name, self.step, self.optguess, self.minbound, self.maxbound


class Uniform(Base):
    """
    A specialization of the Base parameter for uniform distributions
    """

    def __init__(self, name, low, high, *args, **kwargs):
        """
        :name: Name of the parameter
        :low: lower bound of the uniform distribution
        :high: higher bound of the uniform distribution
        :step:     (optional) number for step size required for some algorithms, 
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of 
                rndfunc(*rndargs, size=1000) 
        """
        super(Uniform, self).__init__(name,
                                      rnd.uniform,
                                      (low, high),
                                      *args,
                                      **kwargs)


class List(object):
    """
    A specialization to sample from a list of parameter sets 
    """

    def __init__(self, name, list_of_parametersettings):
        self.icall = 0
        self.name = name
        self.list_of_parametersettings = list_of_parametersettings

    def __call__(self):
        self.icall += 1
        try:
            return self.list_of_parametersettings[self.icall - 3]
        except IndexError:
            text = 'Error: Number of repetitions is higher than the number of available parameter sets'
            print(text)
            raise

    def astuple(self):
        return self(), self.name, 0, 0, 0, 0


class Normal(Base):
    """
    A specialization of the Base parameter for normal distributions
    """

    def __init__(self, name, mean, stddev, *args, **kwargs):
        """
        :name: Name of the parameter
        :mean: center of the normal distribution
        :stddev: variance of the normal distribution
        :step:     (optional) number for step size required for some algorithms, 
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of 
                rndfunc(*rndargs, size=1000) 
        """
        super(Normal, self).__init__(name,
                                     rnd.normal,
                                     (mean, stddev),
                                      *args,
                                     **kwargs)


class logNormal(Base):
    """
    A specialization of the Base parameter for normal distributions
    """

    def __init__(self, name, mean, sigma, *args, **kwargs):
        """
        :name: Name of the parameter
        :mean: Mean value of the underlying normal distribution
        :sigma: Standard deviation of the underlying normal distribution >0
        :step:     (optional) number for step size required for some algorithms, 
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of 
                rndfunc(*rndargs, size=1000) 
        """
        super(logNormal, self).__init__(name,
                                        rnd.lognormal,
                                        (mean, sigma),
                                        *args,
                                        **kwargs)


class Chisquare(Base):
    """
    A specialization of the Base parameter for chisquare distributions
    """

    def __init__(self, name, dt, *args, **kwargs):
        """
        :name: Name of the parameter
        :dt: Number of degrees of freedom.
        :step:     (optional) number for step size required for some algorithms, 
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of 
                rndfunc(*rndargs, size=1000) 
        """
        super(Chisquare, self).__init__(name,
                                        rnd.chisquare,
                                        (dt,),
                                        *args,
                                        **kwargs)


class Exponential(Base):
    """
    A specialization of the Base parameter for exponential distributions
    """

    def __init__(self, name, scale, *args, **kwargs):
        """
        :name: Name of the parameter
        :scale: The scale parameter, \beta = 1/\lambda.
        :step:     (optional) number for step size required for some algorithms, 
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of 
                rndfunc(*rndargs, size=1000) 
        """
        super(Exponential, self).__init__(name,
                                          rnd.exponential,
                                          (scale,),
                                          *args,
                                          **kwargs)


class Gamma(Base):
    """
    A specialization of the Base parameter for gamma distributions
    """

    def __init__(self, name, shape, *args, **kwargs):
        """
        :name: Name of the parameter
        :shape: The shape of the gamma distribution.
        :step:     (optional) number for step size required for some algorithms, 
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of 
                rndfunc(*rndargs, size=1000) 
        """
        super(Gamma, self).__init__(name,
                                    rnd.gamma,
                                    (shape,),
                                    *args,
                                    **kwargs)


class Wald(Base):
    """
    A specialization of the Base parameter for Wald distributions
    """

    def __init__(self, name, mean, scale, *args, **kwargs):
        """
        :name: Name of the parameter
        :mean: Shape of the distribution.
        :scale: Shape of the distribution.
        :step:     (optional) number for step size required for some algorithms, 
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of 
                rndfunc(*rndargs, size=1000) 
        """
        super(Wald, self).__init__(name,
                                   rnd.wald,
                                   (mean, scale),
                                   *args,
                                   **kwargs)


class Weibull(Base):
    """
    A specialization of the Base parameter for Weibull distributions
    """

    def __init__(self, name, a, *args, **kwargs):
        """
        :name: Name of the parameter
        :a: Shape of the distribution.
        :step:     (optional) number for step size required for some algorithms, 
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of 
                rndfunc(*rndargs, size=1000) 
        """
        super(Weibull, self).__init__(name,
                                       rnd.weibull,
                                       (a,),
                                       *args,
                                       **kwargs)


def generate(parameters):
    """
    This function generates a parameter set from a list of parameter objects. The parameter set
    is given as a structured array in the format the parameters function of a setup expects
    :parameters: A sequence of parameter objects
    """
    dtype = [('random', '<f8'), ('name', '|S30'),
             ('step', '<f8'), ('optguess', '<f8'),
             ('minbound', '<f8'), ('maxbound', '<f8')]
    return np.fromiter((param.astuple() for param in parameters), dtype=dtype, count=len(parameters))