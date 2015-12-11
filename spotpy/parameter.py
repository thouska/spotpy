# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Philipp Kraft and Tobias Houska

Contains classes to generate random parameter sets
'''

import numpy.random as rnd
import numpy as np

class Base(object):
    """
    This is a universal random parameter class
    
    
    TODO: Better documentation
    """
    def __init__(self,name,rndfunc,rndargs,step,optguess):
        """
        :name: Name of the parameter
        :rndfunc: Function to draw a random number, eg. the random functions from numpy.random
        :rndargs: Argument-tuple for the random function, eg. lower and higher bound 
                  (number and meaning of arguments depends on the function)
        :step:  Some algorithms, eg. mcmc need a parameter of the variance for the next step
        :optguess: Some algorithms depend on a good start point, this is given by optguess

        """
        self.name =name
        self.rndfunc = rndfunc
        self.rndargs = rndargs
        self.step = step
        self.optguess = optguess
    def __call__(self):
        """
        Returns a pparameter realization
        """
        return self.rndfunc(*self.rndargs)
    def astuple(self):
        """
        Returns a tuple of a realization and the other parameter properties
        """
        return self(),self.name,self.step,self.optguess

class Uniform(Base):
    """
    A specialization of the Base parameter for uniform distributions
    
    """
    def __init__(self,name,low,high,step=None,optguess=None):
        """
        :name: Name of the parameter
        :low: lower bound of the uniform distribution
        :high: higher bound of the uniform distribution
        :step: Step size for mcmc like alg. If None, 0.1 * (high-low) is used
        :optguess: First guess of optimum for mcmc like alg. If None, mean(low,high) is used 
        """
        Base.__init__(self,name,rnd.uniform,(low,high),None,None)
        self.step = step or 0.1 * (high-low)
        self.optguess = optguess or 0.5*(low+high)

class List(object):
    """
    A specialization to sample from a list of parameter sets 
    
    """
    def __init__(self,name,list_of_parametersettings):
        self.icall=0
        self.name=name
        self.list_of_parametersettings=list_of_parametersettings
    
    def __call__(self):
        self.icall+=1
        try:
            return self.list_of_parametersettings[self.icall-3]
        except IndexError:
            text='Error: Number of repetitions is higher than the number of available parameter sets'
            print(text)
            raise
            
    def astuple(self):
        return self(), self.name, 0, 0        
    
    
class Normal(Base):
    """
    A specialization of the Base parameter for normal distributions
    """
    def __init__(self,name,mean,stddev,step=None,optguess=None):
        """
        :name: Name of the parameter
        :mean: center of the normal distribution
        :stddev: variance of the normal distribution
        :step: Step size for mcmc like alg. If None, 0.1 * (high-low) is used
        :optguess: First guess of optimum for mcmc like alg. If None, mean(low,high) is used 
        """

        Base.__init__(self,name,rnd.normal,(mean,stddev),None,None)
        self.step = step or 0.5 * stddev
        self.optguess = optguess or mean

class logNormal(Base):
    """
    A specialization of the Base parameter for normal distributions
    """
    def __init__(self,name,mean,sigma,step=None,optguess=None):
        """
        :name: Name of the parameter
        :mean: Mean value of the underlying normal distribution
        :sigma: Standard deviation of the underlying normal distribution >0
        :step: Step size for mcmc like alg. If None, 0.1 * (high-low) is used
        :optguess: First guess of optimum for mcmc like alg. If None, mean(low,high) is used 
        """

        Base.__init__(self,name,rnd.lognormal,(mean,sigma),None,None)
        self.step = step or 0.5 * stddev
        self.optguess = optguess or mean
        
class Chisquare(Base):
    """
    A specialization of the Base parameter for chisquare distributions
    """
    def __init__(self,name,dt,step=None,optguess=None):
        """
        :name: Name of the parameter
        :dt: Number of degrees of freedom.
        :step: Step size for mcmc like alg. If None, 0.1 * (high-low) is used
        :optguess: First guess of optimum for mcmc like alg. If None, mean(low,high) is used 
        """

        Base.__init__(self,name,rnd.chisquare,(dt),None,None)
        self.step = step or 0.5 * stddev
        self.optguess = optguess or mean

class Exponential(Base):
    """
    A specialization of the Base parameter for exponential distributions
    """
    def __init__(self,name,scale,step=None,optguess=None):
        """
        :name: Name of the parameter
        :scale: The scale parameter, \beta = 1/\lambda.
        :step: Step size for mcmc like alg. If None, 0.1 * (high-low) is used
        :optguess: First guess of optimum for mcmc like alg. If None, mean(low,high) is used 
        """

        Base.__init__(self,name,rnd.exponential,(scale),None,None)
        self.step = step or 0.5 * stddev
        self.optguess = optguess or mean        

class Gamma(Base):
    """
    A specialization of the Base parameter for gamma distributions
    """
    def __init__(self,name,shape,step=None,optguess=None):
        """
        :name: Name of the parameter
        :shape: The shape of the gamma distribution.
        :step: Step size for mcmc like alg. If None, 0.1 * (high-low) is used
        :optguess: First guess of optimum for mcmc like alg. If None, mean(low,high) is used 
        """

        Base.__init__(self,name,rnd.gamma,(shape),None,None)
        self.step = step or 0.5 * stddev
        self.optguess = optguess or mean
        
class Wald(Base):
    """
    A specialization of the Base parameter for Wald distributions
    """
    def __init__(self,name,mean,scale,step=None,optguess=None):
        """
        :name: Name of the parameter
        :mean: Shape of the distribution.
        :scale: Shape of the distribution.
        :step: Step size for mcmc like alg. If None, 0.1 * (high-low) is used
        :optguess: First guess of optimum for mcmc like alg. If None, mean(low,high) is used 
        """

        Base.__init__(self,name,rnd.wald,(mean,scale),None,None)
        self.step = step or 0.5 * stddev
        self.optguess = optguess or mean
        
        
class Weilbull(Base):
    """
    A specialization of the Base parameter for Weilbull distributions
    """
    def __init__(self,name,a,step=None,optguess=None):
        """
        :name: Name of the parameter
        :a: Shape of the distribution.
        :step: Step size for mcmc like alg. If None, 0.1 * (high-low) is used
        :optguess: First guess of optimum for mcmc like alg. If None, mean(low,high) is used 
        """

        Base.__init__(self,name,rnd.weilbull,(a),None,None)
        self.step = step or 0.5 * stddev
        self.optguess = optguess or mean

        
def generate(parameters):
    """
    This function generates a parameter set from a list of parameter objects. The parameter set
    is given as a structured array in the format the parameters function of a setup expects
    
    :parameters: A sequence of parameter objects
    """
    dtype=[('random', '<f8'), ('name', '|S30'),('step', '<f8'),('optguess', '<f8')]
    return np.fromiter((param.astuple() for param in parameters),dtype=dtype,count=len(parameters))
        