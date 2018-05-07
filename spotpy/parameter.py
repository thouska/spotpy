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

if sys.version_info[0] >= 3:
    unicode = str

from collections import namedtuple
from itertools import cycle


class _ArgumentHelper(object):
    """
    A helper to assess the arguments to the __init__ method of a parameter.

    Using constructor arguments in Python is normally trivial. However, with the spotpy parameters,
    we are between a rock and a hard place:
    On the one hand, the standard way to create a parameter should not change, and that means
    the first argument to the __init__ method of a parameter is the name. On the other hand
    the new way to define parameters as class properties in a setup class is ugly if a parameter name
    needs to be given. This class helps by checking keyword and arguments for their types.
    """
    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
        self.classname = type(parent).__name__
        self.args = list(args)
        self.kwargs = kwargs.copy()
        self.processed_args = 0


    def name(self):
        """
        A helper method for Base.__init__.

        Looks for a name of the parameter.
        First it looks at args[0], if this is a string, this function assumes it is the name and the
        distribution arguments follow. If args[0] is not a string but a number, the function looks
        for a keyword argument "name" and uses that or, if it fails the name of the parameter is the
        empty string

        For the usage of this function look at the parameter realisations in this file, eg. Uniform

        :return: name
        """
        # Check if args[0] is string like (and exists)
        if self.args and unicode(self.args[0]) == self.args[0]:
            name = self.args.pop(0)
            self.processed_args += 1
        # else get the name from the keywords
        elif 'name' in self.kwargs:
            name = self.kwargs.pop('name')
            self.processed_args += 1
        # or just do not use a name
        else:
            name = ''

        return name

    def alias(self, name, target):
        """
        Moves a keyword from one name to another
        """
        if name in self.kwargs and target not in self.kwargs:
            self.kwargs[target] = self.kwargs.pop(name)

    def attributes(self, names, raise_for_missing=None, as_dict=False):
        """

        :param names:
        :param raise_for_missing:
        :return:
        """
        # A list of distribution parameters
        attributes = []
        # a list of distribution parameter names that are missing. Should be empty.
        missing = []

        # Loop through the parameter names to get their values from
        # a) the args
        # b) if the args are fully processed, the kwargs
        # c) if the args are processed and the name is not present in the kwargs, add to missing
        for i, pname in enumerate(names):
            if self.args:
                # First use up positional arguments for the rndargs
                attributes.append(self.args.pop(0))
                self.processed_args += 1
            elif pname in self.kwargs:
                # If no positional arguments are left, look for a keyword argument
                attributes.append(self.kwargs.pop(pname))
                self.processed_args += 1
            else:
                # Argument not found, add to missing and raise an Exception afterwards
                missing.append(pname)
                attributes.append(None)

        # If the algorithm did not find values for distribution parameters in args are kwargs, fail
        if missing and raise_for_missing:
            raise TypeError(
                '{T} expected values for the parameters {m}'.format(
                    T=self.classname,
                    m=', '.join(missing)
                )
            )
        # Return the name, the distribution parameter values, and a tuple of unprocessed args and kwargs
        if as_dict:
            return dict((n, a) for n, a in zip(names, attributes))
        else:
            return attributes

    def __len__(self):
        return len(self.args) + len(self.kwargs)

    def get(self, argname):
        """
        Checks if argname is in kwargs, if present it is returned and removed else none.
        :param argname:
        :return:
        """
        return self.kwargs.pop(argname, None)

    def check_complete(self):
        """
        Checks if all args and kwargs have been processed.
        Raises TypeError if unprocessed arguments are left
        """
        total_args = len(self) + self.processed_args
        if len(self):
            error = '{}: {} arguments where given but only {} could be used'.format(self.classname, total_args, self.processed_args)
            raise TypeError(error)

def _round_sig(x, sig=3):
    from math import floor, log10
    return round(x, sig-int(floor(log10(abs(x))))-1)

class Base(object):
    """
    This is a universal random parameter class
    It creates a random number (or array) drawn from specified distribution.

    How to create a concrete Parameter class:

    Let us assume, we have a random distribution function foo with the parameters a and b:
    ``foo(a, b, size=1000)``. Then the parameter class is coded as:

    .. code ::
        class Foo(Base):
            __rndargs__ = 'a', 'b' # A tuple of the distribution argument names
            def __init__(*args, **kwargs):
                Base.__init__(foo, *args, **kwargs)

    The Uniform parameter class is the reference implementation.
    """
    __rndargs__ = ()
    def __init__(self, rndfunc, *args, **kwargs):
        """
        :name:     Name of the parameter
        :rndfunc:  Function to draw a random number, 
                   eg. the random functions from numpy.random using the rndargs
        :rndargs:  tuple of the argument names for the random function
                   eg. for uniform: ('low', 'high'). The values for the rndargs are retrieved
                   from positional and keyword arguments, args and kwargs.
        :step:     (optional) number for step size required for some algorithms
                    eg. mcmc need a parameter of the variance for the next step
                    default is quantile(0.5) - quantile(0.4) of
        :optguess: (optional) number for start point of parameter
                default is median of rndfunc(*rndargs, size=1000)
                rndfunc(*rndargs, size=1000)
        """
        self.rndfunc = rndfunc
        arghelper = _ArgumentHelper(self, *args, **kwargs)
        self.name = arghelper.name()
        arghelper.alias('default', 'optguess')
        self.rndargs = arghelper.attributes(type(self).__rndargs__, type(self).__name__)

        if self.rndfunc:
            # Get the standard arguments for the parameter or create them
            param_args = arghelper.attributes(['step', 'optguess', 'minbound', 'maxbound'], as_dict=True)
            # Draw one sample of size 1000
            sample = self(size=1000)
            self.step = param_args.get('step') or (np.percentile(sample, 50) - np.percentile(sample, 40))
            self.optguess = param_args.get('optguess') or np.median(sample)
            self.minbound = param_args.get('minbound') or _round_sig(np.min(sample))
            self.maxbound = param_args.get('maxbound') or _round_sig(np.max(sample))

        else:

            self.step = 0.0
            self.optguess = 0.0
            self.minbound = 0.0
            self.maxbound = 0.0

        self.description = arghelper.get('doc')
        arghelper.check_complete()

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
        doc = vars(self).get('description')
        if doc:
            res = '{} ({})'.format(doc, repr(self))
            if sys.version_info.major == 2:
                return res.encode('utf-8', errors='ignore')
            else:
                return res
        else:
            return repr(self)

    def __unicode__(self):
        doc = vars(self).get('description')
        if doc:
            return u'{}({})'.format(unicode(doc), repr(self))
        else:
            return repr(self)



class Uniform(Base):
    """
    A specialization of the Base parameter for uniform distributions
    """
    __rndargs__ = 'low', 'high'
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
        super(Uniform, self).__init__(rnd.uniform, *args, **kwargs)


class List(Base):
    """
    A specialization to sample from a list (or other iterable) of parameter sets.

    Usage:
    >>>list_param = List([1,2,3,4], repeat=True)
    >>>list_param()
    1
    """
    __rndargs__ = ('values', )
    def __init__(self, *args, **kwargs):
        self.repeat = kwargs.pop('repeat', False)
        super(List, self).__init__(None,  *args, **kwargs)
        self.values, = self.rndargs

        # Hack to avoid skipping the first value. See __call__ function below.
        self.throwaway_first = True

        if self.repeat:
            # If the parameter list should repeated, create an inifinite loop of the data iterator
            self.iterator = cycle(self.values)
        else:
            # If the list should end when the list is exhausted, just get a normal iterator
            self.iterator = iter(self.values)

    def __call__(self, size=None):
        """
        Returns the next value from the data list
        :param size: Number of sample to draw from data
        :return:
        """
        # Hack to avoid skipping the first value of the parameter list.
        # This function is called once when the _algorithm __init__
        # has to initialize the parameter names. Because of this, we end up
        # losing the first value in the list, which is undesirable
        # This check makes sure that the first call results in a dummy value
        if self.throwaway_first:
            self.throwaway_first = False
            return None

        if size:
            return np.fromiter(self.iterator, dtype=float, count=size)
        else:
            try:
                return next(self.iterator)
            except StopIteration:
                text = 'Number of repetitions is higher than the number of available parameter sets'
                raise IndexError(text)

    def astuple(self):
        return self(), self.name, 0, 0, 0, 0


class Constant(Base):
    """
    A specialization that produces always the same constant value
    """
    __rndargs__ = 'scalar',

    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(self, *args, **kwargs)

    def __call__(self, size=None):
        """
        Returns the next value from the data list
        :param size: Number of items to draw from parameter
        :return:
        """
        if size:
            return np.ones(size, dtype=float) * self.rndargs[0]
        else:
            return self.rndargs[0]

    def astuple(self):
        return self(), self.name, self.rndargs[0], self.rndargs[0], 0, 0


class Normal(Base):
    """
    A specialization of the Base parameter for normal distributions
    """
    __rndargs__ = 'mean', 'stddev'
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

        super(Normal, self).__init__(rnd.normal, *args, **kwargs)


class logNormal(Base):
    """
    A specialization of the Base parameter for normal distributions
    """
    __rndargs__ = 'mean', 'sigma'
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
        super(logNormal, self).__init__(rnd.lognormal, *args, **kwargs)


class Chisquare(Base):
    """
    A specialization of the Base parameter for chisquare distributions
    """
    __rndargs__ = 'dt',
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
        super(Chisquare, self).__init__(rnd.chisquare, *args, **kwargs)


class Exponential(Base):
    """
    A specialization of the Base parameter for exponential distributions
    """
    __rndargs__ = 'scale',

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
        super(Exponential, self).__init__(rnd.exponential,  *args, **kwargs)


class Gamma(Base):
    """
    A specialization of the Base parameter for gamma distributions
    """
    __rndargs__ = 'shape', 'scale'

    def __init__(self, *args, **kwargs):
        """
        :name: Name of the parameter
        :shape: The shape of the gamma distribution.
        :scale: The scale of the gamme distribution
        :step:     (optional) number for step size required for some algorithms,
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of
                rndfunc(*rndargs, size=1000)
        """

        super(Gamma, self).__init__(rnd.gamma,  *args, **kwargs)


class Wald(Base):
    """
    A specialization of the Base parameter for Wald distributions
    """

    __rndargs__ = 'mean', 'scale'
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
        super(Wald, self).__init__(rnd.wald, *args, **kwargs)


class Weibull(Base):
    """
    A specialization of the Base parameter for Weibull distributions
    """
    __rndargs__ = 'a',
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
        super(Weibull, self).__init__(rnd.weibull, *args, **kwargs)

class Triangular(Base):
    """
    A parameter sampling from a triangular distribution
    """
    __rndargs__ = 'left', 'mode', 'right'
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
        super(Triangular, self).__init__(rnd.triangular, *args, **kwargs)


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


def get_parameters_array(setup):
    """
    Returns the parameter array from the setup
    """
    # Put the parameter arrays as needed here, they will be merged at the end of this
    # function
    param_arrays = []
    # Get parameters defined with the setup class
    param_arrays.append(
        # generate creates the array as defined in the setup API
        generate(get_parameters_from_setup(setup))
    )

    if hasattr(setup, 'parameters') and callable(setup.parameters):
        # parameters is a function, as defined in the setup API up to at least spotpy version 1.3.13
        param_arrays.append(setup.parameters())

    # Return the class and the object parameters together
    res = np.concatenate(param_arrays)
    return res


def create_set(setup, valuetype='optguess', **kwargs):
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
    :param valuetype: Select between 'optguess' (defaul), 'random', 'minbound' and 'maxbound'.
    :param kwargs: Any keywords can be used to set certain parameters to fixed values
    :return: namedtuple of parameter values
    """

    # Get the array of parameter realizations
    params = get_parameters_array(setup)

    # Create the namedtuple from the parameter names
    partype = get_namedtuple_from_paramnames(setup, params['name'])

    # Use the generated values from the distribution
    pardict = dict(zip(params['name'], params[valuetype]))

    # Overwrite parameters with keyword arguments
    pardict.update(kwargs)

    # Return the namedtuple with fitting names
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
    parnames = ["p" + x if x.isdigit() else x for x in list(parnames)]
    return namedtuple('Par_' + typename,  # Type name created from the setup name
                      parnames)  # get parameter names


def get_parameters_from_setup(setup):
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
    >>>
    >>> setup = SpotpySetup()
    >>> parameters = spotpy.parameter.get_parameters_from_setup(setup)
    """

    # Get all class variables
    cls = type(setup)
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

    # Get the parameters list of setup if the parameter attribute is not a method:
    if hasattr(setup, 'parameters') and not callable(setup.parameters):
        # parameters is not callable, assume a list of parameter objects.
        # Generate the parameters array from it and append it to our list
        parameters.extend(setup.parameters)

    return parameters


if __name__ == '__main__':
    class TestSetup:
        a = Uniform(0, 1)
        b = Triangular(0, 1, right=10, doc='A Triangular parameter')

        def __init__(self):
            self.parameters = [Uniform('c', 0, 1), Uniform('d', 0, 2)]

    params = get_parameters_from_setup(TestSetup)
    assert len(params) == 4, 'Expected 2 Parameters in the test setup, got {}'.format(len(params))
    assert params[0] is TestSetup.a, 'Expected parameter a to be first in params'
    assert params[1] is TestSetup.b
    print('\n'.join(str(p) for p in params))

