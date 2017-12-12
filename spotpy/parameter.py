# -*- coding: utf-8 -*-
'''
Copyright (c) 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Philipp Kraft and Tobias Houska
Contains classes to generate random parameter sets
'''
from __future__ import division, print_function, absolute_import
import numpy.random as rnd
import numpy as np
import sys

if sys.version_info.major == 3:
    unicode = str

from collections import namedtuple
from itertools import cycle
from inspect import cleandoc

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
        if self.rndfunc:
            self.step = step or np.percentile(self(size=1000), 50)
            self.optguess = optguess or (np.percentile(self(size=1000), 50) -
                                         np.percentile(self(size=1000), 40))
            self.minbound = minbound or np.min(self(size=1000))
            self.maxbound = maxbound or np.max(self(size=1000))
        else:
            self.step = 0.0
            self.optguess = 0.0
            self.minbound = 0.0
            self.maxbound = 0.0
        self.description = kwargs.get('doc')


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
    
    def __repr__(self):
        """
        Returns a textual representation of the parameter
        """
        return "{tname}('{p.name}', {p.rndargs})".format(tname=type(self).__name__, p=self)

    def __str__(self):
        """
        Returns the description of the parameter, if available, else repr(self)
        :return: 
        """
        if vars(self).get('description'):
            return '{} ({})'.format(vars(self).get('description'), repr(self))
        else:
            return repr(self)

    def _get_name_from_args(self, params, *args, **kwargs):
        """
        Gets the name from the otherwise float arguments
        If is args[0] is a string, return args[0] as the name else return '' or kwargs['name'] if present.
        The other parameters of the distribution, (given as the params string) are deduced from args and kwargs.
        The returned args and kwargs are without the distribution parameters

        For the usage of this function look at the parameter realisations in this file, eg. Uniform
        
        :param params: A space seperated string of the expected parameter names of the distribution
        :param args: An argument tuple given to a parameter constructor
        :param kwargs: The keyword arguments
        :return: name, distributionparam1, distributionparam2, remaining_args, remaining_kwargs
        """
        args = list(args) # Make args mutable
        # Check if args[0] is string like
        if unicode(args[0]) == args[0]:
            name = args.pop(0)
        # else get the name from the keywords
        elif 'name' in kwargs:
            name = kwargs.pop('name')
        # or just do not use a name
        else:
            name = ''

        # A list of distribution parameters and a list of distribution parameter names that are missing
        plist = []
        missing = []

        # Loop through the parameter names to get their values from
        # a) the args
        # b) if the args are fully processed, the kwargs
        # c) if the args are processed and the name is not present in the kwargs, add to missing
        for i, pname in enumerate(params.split()):
            if args:
                plist.append(args.pop(0))
            elif pname in kwargs:
                plist.append(kwargs.pop(pname))
            else:
                missing.append(pname)
                plist.append(None)

        # If the algo did not find values for distribution parameters in args are kwargs, fail
        if missing:
            raise ValueError(
                '{T} expected values for the parameters {m}'.format(
                    T=type(self).__name__,
                    m=', '.join(missing)
                )
            )
        # Return the name, the distribution parameter values, and a tuple of unprocessed args and kwargs
        return (name,) + tuple(plist) + (tuple(args), kwargs)


class Uniform(Base):
    """
    A specialization of the Base parameter for uniform distributions
    """

    def __init__(self, *args, **kwargs):
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
        name, low, high, args, kwargs = self._get_name_from_args('low high', *args, **kwargs)
        super(Uniform, self).__init__(name,
                                      rnd.uniform,
                                      (low, high),
                                      *args,
                                      **kwargs)


class List(Base):
    """
    A specialization to sample from a list (or other iterable) of parameter sets.

    Usage:
    >>>list_param = List([1,2,3,4], repeat=True)
    >>>list_param()
    1
    """

    def __init__(self, *args, **kwargs):
        name, list_of_parametersettings, args, kwargs = self._get_name_from_args('list_of_parametersettings', *args, **kwargs)
        super().__init__(name, None, None, None, None, None, *args, **kwargs)
        self.name = name
        self.repeat = kwargs.get('repeat', False)

        if self.repeat:
            # If the parameter list should repeated, create an inifinite loop of the data iterator
            self.iterator = cycle(list_of_parametersettings)
        else:
            # If the list should end when the list is exhausted, just get a normal iterator
            self.iterator = iter(list_of_parametersettings)

    def __call__(self, size=None):
        """
        Returns the next value from the data list
        :param size: Number of sample to draw from data
        :return:
        """
        if size:
            return np.fromiter(self.iterator, dtype=float, count=size)
        else:
            try:
                return next(self.iterator)
            except StopIteration:
                text = 'Error: Number of repetitions is higher than the number of available parameter sets'
                print(text)
                raise

    def astuple(self):
        return self(), self.name, 0, 0, 0, 0


class Constant(Base):
    """
    A specialization that produces always the same constant value
    """

    def __init__(self, *args, **kwargs):
        name, scalar, args, kwargs = self._get_name_from_args('list_of_parametersettings', *args, **kwargs)
        super().__init__(name, None, None, None, None, None, *args, **kwargs)
        self.scalar = scalar

    def __call__(self, size=None):
        """
        Returns the next value from the data list
        :param size: Number of items to draw from parameter
        :return:
        """
        if size:
            return np.ones(size, dtype=float) * self.scalar
        else:
            return self.scalar

    def astuple(self):
        return self(), self.name, self.scalar, self.scalar, 0, 0


class Normal(Base):
    """
    A specialization of the Base parameter for normal distributions
    """

    def __init__(self, *args, **kwargs):
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
        name, mean, stddev, args, kwargs = self._get_name_from_args('mean stddev', *args, **kwargs)

        super(Normal, self).__init__(name,
                                     rnd.normal,
                                     (mean, stddev),
                                     *args,
                                     **kwargs)


class logNormal(Base):
    """
    A specialization of the Base parameter for normal distributions
    """

    def __init__(self, *args, **kwargs):
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
        name, mean, sigma, args, kwargs = self._get_name_from_args('mean sigma', *args, **kwargs)
        super(logNormal, self).__init__(name,
                                        rnd.lognormal,
                                        (mean, sigma),
                                        *args,
                                        **kwargs)


class Chisquare(Base):
    """
    A specialization of the Base parameter for chisquare distributions
    """

    def __init__(self, *args, **kwargs):
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
        name, dt, args, kwargs = self._get_name_from_args('dt', *args, **kwargs)
        super(Chisquare, self).__init__(name,
                                        rnd.chisquare,
                                        (dt,),
                                        *args,
                                        **kwargs)


class Exponential(Base):
    """
    A specialization of the Base parameter for exponential distributions
    """

    def __init__(self, *args, **kwargs):
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
        name, mean, scale, args, kwargs = self._get_name_from_args('mean scale', *args, **kwargs)
        super(Exponential, self).__init__(name,
                                          rnd.exponential,
                                          (scale,),
                                          *args,
                                          **kwargs)


class Gamma(Base):
    """
    A specialization of the Base parameter for gamma distributions
    """

    def __init__(self, *args, **kwargs):
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
        name, shape, args, kwargs = self._get_name_from_args('shape', *args, **kwargs)

        super(Gamma, self).__init__(name,
                                    rnd.gamma,
                                    (shape,),
                                    *args,
                                    **kwargs)


class Wald(Base):
    """
    A specialization of the Base parameter for Wald distributions
    """

    def __init__(self, *args, **kwargs):
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
        name, mean, scale, args, kwargs = self._get_name_from_args('mean scale', *args, **kwargs)

        super(Wald, self).__init__(name,
                                   rnd.wald,
                                   (mean, scale),
                                   *args,
                                   **kwargs)


class Weibull(Base):
    """
    A specialization of the Base parameter for Weibull distributions
    """

    def __init__(self, *args, **kwargs):
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
        name, a, args, kwargs = self._get_name_from_args('a', *args, **kwargs)

        super(Weibull, self).__init__(name,
                                      rnd.weibull,
                                      (a,),
                                      *args,
                                      **kwargs)

class Triangular(Base):
    """
    A parameter sampling from a triangular distribution
    """
    def __init__(self, *args, **kwargs):
        """
        :name: Name of the parameter
        :left: Lower limit of the parameter
        :mode: The value where the peak of the distribution occurs.
        :right: Upper limit, should be larger than `left`.
        :step:     (optional) number for step size required for some algorithms, 
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of 
                rndfunc(*rndargs, size=1000) 
        """
        name, left, mode, right, args, kwargs = self._get_name_from_args('left mode right', *args, **kwargs)
        super(Triangular, self).__init__(name,
                                         rnd.triangular,
                                         (left, mode, right),
                                         *args,
                                         **kwargs)


def generate(parameters):
    """
    This function generates a parameter set from a list of parameter objects. The parameter set
    is given as a structured array in the format the parameters function of a setup expects
    :parameters: A sequence of parameter objects
    """
    dtype = [('random', '<f8'), ('name', '|U30'),
             ('step', '<f8'), ('optguess', '<f8'),
             ('minbound', '<f8'), ('maxbound', '<f8')]

    return np.fromiter((param.astuple() for param in parameters), dtype=dtype, count=len(parameters))


def get_parameters_from_setup(setup):
    """
    Returns the parameter array from the setup
    """

    # Put the parameter arrays as needed here, they will be merged at the end of this
    # function
    param_arrays = []
    # Get parameters defined with the setup class
    param_arrays.append(
        # generate creates the array as defined in the setup API
        generate(get_parameters_from_class(type(setup)))
    )

    if hasattr(setup, 'parameters'):
        # object parameters, the old school way to create parameters
        if callable(setup.parameters):
            # parameters is a function, as defined in the setup API up to at least spotpy version 1.3.13
            param_arrays.append(setup.parameters())
        else:
            # parameters is not callable, assume a list of parameter objects.
            # Generate the parameters array from it and append it to our list
            param_arrays.append(generate(setup.parameters))

    # Return the class and the object parameters together
    return np.concatenate(param_arrays)

def describe_setup(setup):
    """
    Describes a spotpy setup using its class name, docstring and parameters
    :param setup: A spotpy compatible model setup
    :return: A describing string
    """

    # Get class name
    s = unicode(type(setup).__name__)
    # Add hbar
    s += '\n' + 30 * '-' + '\n\n'

    # Add doc string
    if sys.version_info.major >= 3:
        mdoc = cleandoc(setup.__doc__ or '')
    else:
        mdoc = setup.__doc__ or ''

    s += mdoc + '\n'

    # Get parameters from class
    params = get_parameters_from_class(type(setup))

    # Get parameters from setup.parameters if it is not a function
    if hasattr(setup, 'parameter') and not callable(setup.parameters):
        try:
            params += list(setup.parameters)
        except TypeError:
            raise ValueError('setup.parameter must be either callable or list of parameters')

    params = '\n'.join(' - {p}'.format(p=p) for p in params)

    # Add parameters to string
    s += '\n'
    s += 'Parameters:\n{params}'.format(params=params)

    return s

def create_set(setup, random=False, **kwargs):
    """
    Returns a named tuple holding parameter values, to be used with the simulation method of a setup

    This function is typically used to test models, before they are used in a sampling

    Usage:
    >>> import spotpy
    >>> from spotpy.examples.spot_setup_rosenbrock import spot_setup
    >>> model = spot_setup()
    >>> param_set = spotpy.parameter.create_set(model, x=2)
    >>> result = model.simulation(param_set)

    :param setup: A spotpy compatible Model object
    :param random: If True, undefined parameters are filled with a random realisation of the
                    parameter distribution, when False undefined parameters are filled with optguess
    :param kwargs: Any keywords can be used to set certain parameters to fixed values
    :return: namedtuple of parameter values
    """
    params = get_parameters_from_setup(setup)
    partype = get_namedtuple_from_paramnames(type(setup).__name__, params['name'])
    if random:
        pardict = dict(zip(params['name'], params['random']))
    else:
        pardict = dict(zip(params['name'], params['optguess']))

    pardict.update(kwargs)

    return partype(**pardict)


def get_namedtuple_from_paramnames(owner, parnames):
    """
    Returns the namedtuple classname for parameter names
    :param owner: Owner of the parameters, usually the spotpy setup
    :param parnames: Sequence of parameter names
    :return: Class
    """

    # Get name of owner class
    typename = type(owner).__name__
    return namedtuple('Par_' + typename,  # Type name created from the setup name
                      list(parnames))  # get parameter names

def get_parameters_from_class(cls):
    """
    Returns a list of the class defined parameters, and
    overwrites the names of the parameters. 
    By defining parameters as class members, as shown below,
    one can omit the parameters function of the setup.
    
    Usage:
    
    >>> from spotpy import parameter
    >>> class SpotpySetup:
    >>>     # Add parameters p1 & p2 to the setup. 
    >>>     p1 = parameter.Uniform(20, 100)
    >>>     p2 = parameter.Gamma(2.3)

    """

    # Get all class variables
    class_variables = vars(cls).items()

    parameters = []
    for attrname, attrobj in class_variables:
        # Check if it is an spotpy parameter
        if isinstance(attrobj, Base):
            # Set the attribute name
            if not attrobj.name:
                attrobj.name = attrname
            # Add parameter to dict
            parameters.append(attrobj)

    # starting with Python 3.6 the order of the class defined parameters are presevered with vars,
    # prior the sorting changes.
    # For a MPI parallized run with spotpy, this is terrible, since the spotpy code expects
    # the list of parameters to be ordered
    # For older pythons, we will sort the list of parameters by their name. For >=3.6, we keep
    # the order.
    if sys.version_info[:3] < (3, 6, 0):
        # Sort the list parameters by name
        parameters.sort(key=lambda p: p.name)

    return parameters


if __name__ == '__main__':
    class TestSetup:
        a = Uniform(0, 1)
        b = Triangular(0, 1, right=10, doc='A Triangular parameter')

    params = get_parameters_from_class(TestSetup)
    assert len(params) == 2, 'Expected 2 Parameters in the test setup, got {}'.format(len(params))
    assert params[0] is TestSetup.a, 'Expected parameter a to be first in params'
    assert params[1] is TestSetup.b
    print('\n'.join(str(p) for p in params))

