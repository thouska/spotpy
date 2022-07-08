# -*- coding: utf-8 -*-
"""
Copyright (c) 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Philipp Kraft and Tobias Houska
Contains classes to generate random parameter sets
"""
import copy
import sys
from itertools import cycle

import numpy as np
import numpy.random as rnd


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
        if self.args and str(self.args[0]) == self.args[0]:
            name = self.args.pop(0)
            self.processed_args += 1
        # else get the name from the keywords
        elif "name" in self.kwargs:
            name = self.kwargs.pop("name")
            self.processed_args += 1
        # or just do not use a name
        else:
            name = ""

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
                "{T} expected values for the parameters {m}".format(
                    T=self.classname, m=", ".join(missing)
                )
            )
        # Return the name, the distribution parameter values, and a tuple of unprocessed args and kwargs
        if as_dict:
            # Creates the name / value dict with entries where the value is not None
            return dict((n, a) for n, a in zip(names, attributes) if a is not None)
        else:
            return attributes

    def __len__(self):
        return len(self.args) + len(self.kwargs)

    def get(self, argname, default=None):
        """
        Checks if argname is in kwargs, if present it is returned and removed else none.
        :param argname:
        :return:
        """
        return self.kwargs.pop(argname, default)

    def check_complete(self):
        """
        Checks if all args and kwargs have been processed.
        Raises TypeError if unprocessed arguments are left
        """
        total_args = len(self) + self.processed_args
        if len(self):
            error = "{}: {} arguments where given but only {} could be used".format(
                self.classname, total_args, self.processed_args
            )
            raise TypeError(error)


def _round_sig(x, sig=3):
    """
    Rounds x to sig significant digits
    :param x: The value to round
    :param sig: Number of significant digits
    :return: rounded value
    """
    from math import floor, log10

    # Check for zero to avoid math value error with log10(0.0)
    if abs(x) < 1e-12:
        return 0
    else:
        return round(x, sig - int(floor(log10(abs(x)))) - 1)


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

    def __init__(self, rndfunc, rndfuncname, *args, **kwargs):
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
        self.rndfunctype = rndfuncname
        arghelper = _ArgumentHelper(self, *args, **kwargs)
        self.name = arghelper.name()
        arghelper.alias("default", "optguess")
        self.rndargs = arghelper.attributes(type(self).__rndargs__, type(self).__name__)

        if self.rndfunc:
            # Get the standard arguments for the parameter or create them
            param_args = arghelper.attributes(
                ["step", "optguess", "minbound", "maxbound"], as_dict=True
            )
            # Draw one sample of size 1000
            sample = self(size=1000)
            self.step = param_args.get(
                "step",
                _round_sig(np.percentile(sample, 50) - np.percentile(sample, 40)),
            )
            self.optguess = param_args.get("optguess", _round_sig(np.median(sample)))
            self.minbound = param_args.get("minbound", _round_sig(np.min(sample)))
            self.maxbound = param_args.get("maxbound", _round_sig(np.max(sample)))

        else:

            self.step = 0.0
            self.optguess = 0.0
            self.minbound = 0.0
            self.maxbound = 0.0

        self.description = arghelper.get("doc")

        self.as_int = not not arghelper.get("as_int")
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
        return (
            self(),
            self.name,
            self.step,
            self.optguess,
            self.minbound,
            self.maxbound,
            self.as_int,
        )

    def __repr__(self):
        """
        Returns a textual representation of the parameter
        """
        return "{tname}('{p.name}', {p.rndargs})".format(
            tname=type(self).__name__, p=self
        )

    def __str__(self):
        """
        Returns the description of the parameter, if available, else repr(self)
        :return:
        """
        doc = vars(self).get("description")
        if doc:
            res = "{} ({})".format(doc, repr(self))
            return res
        else:
            return repr(self)

    def __unicode__(self):
        doc = vars(self).get("description")
        if doc:
            return "{}({})".format(str(doc), repr(self))
        else:
            return repr(self)


class Uniform(Base):
    """
    A specialization of the Base parameter for uniform distributions
    """

    __rndargs__ = "low", "high"

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
        super(Uniform, self).__init__(rnd.uniform, "Uniform", *args, **kwargs)


class List(Base):
    """
    A specialization to sample from a list (or other iterable) of parameter sets.

    Usage:
    list_param = List([1,2,3,4], repeat=True)
    list_param()
    1
    """

    __rndargs__ = ("values",)

    def __init__(self, *args, **kwargs):
        self.repeat = kwargs.pop("repeat", False)
        super(List, self).__init__(None, "List", *args, **kwargs)
        (self.values,) = self.rndargs

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
                text = "Number of repetitions is higher than the number of available parameter sets"
                raise IndexError(text)

    def astuple(self):
        return self(), self.name, 0, 0, 0, 0, self.as_int


class Constant(Base):
    """
    A specialization that produces always the same constant value
    """

    __rndargs__ = ("scalar",)

    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(self, "Constant", *args, **kwargs)

    value = property(lambda self: self.rndargs[0])

    def __call__(self, size=None):
        """
        Returns the next value from the data list
        :param size: Number of items to draw from parameter
        :return:
        """
        if size:
            return np.ones(size, dtype=float) * self.value
        else:
            return self.value

    def astuple(self):
        return self(), self.name, 0, self.value, self.value, self.value, self.as_int


class Normal(Base):
    """
    A specialization of the Base parameter for normal distributions
    """

    __rndargs__ = "mean", "stddev"

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

        super(Normal, self).__init__(rnd.normal, "Normal", *args, **kwargs)


class logNormal(Base):
    """
    A specialization of the Base parameter for normal distributions
    """

    __rndargs__ = "mean", "sigma"

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
        super(logNormal, self).__init__(rnd.lognormal, "logNormal", *args, **kwargs)


class Chisquare(Base):
    """
    A specialization of the Base parameter for chisquare distributions
    """

    __rndargs__ = ("dt",)

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
        super(Chisquare, self).__init__(rnd.chisquare, "Chisquare", *args, **kwargs)


class Exponential(Base):
    """
    A specialization of the Base parameter for exponential distributions
    """

    __rndargs__ = ("scale",)

    def __init__(self, *args, **kwargs):
        """
        :name: Name of the parameter
        :scale: The scale parameter, \beta = 1 divided by lambda.
        :step:     (optional) number for step size required for some algorithms,
                eg. mcmc need a parameter of the variance for the next step
                default is median of rndfunc(*rndargs, size=1000)
        :optguess: (optional) number for start point of parameter
                default is quantile(0.5) - quantile(0.4) of
                rndfunc(*rndargs, size=1000)
        """
        super(Exponential, self).__init__(
            rnd.exponential, "Exponential", *args, **kwargs
        )


class Gamma(Base):
    """
    A specialization of the Base parameter for gamma distributions
    """

    __rndargs__ = "shape", "scale"

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

        super(Gamma, self).__init__(rnd.gamma, "Gamma", *args, **kwargs)


class Wald(Base):
    """
    A specialization of the Base parameter for Wald distributions
    """

    __rndargs__ = "mean", "scale"

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
        super(Wald, self).__init__(rnd.wald, "Wald", *args, **kwargs)


class Weibull(Base):
    """
    A specialization of the Base parameter for Weibull distributions
    """

    __rndargs__ = ("a",)

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
        super(Weibull, self).__init__(rnd.weibull, "Weibull", *args, **kwargs)


class Triangular(Base):
    """
    A parameter sampling from a triangular distribution
    """

    __rndargs__ = "left", "mode", "right"

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
        super(Triangular, self).__init__(rnd.triangular, "Triangular", *args, **kwargs)


class ParameterSet(object):
    """
    A Pickable parameter set to use named parameters in a setup
    Is not created by a user directly, but in algorithm.
    Older versions used a namedtuple, which is not pickable.

    An instance of ParameterSet is sent to the users setup.simulate method.

    Usage:
     ps = ParameterSet(...)

    Update values by arguments or keyword arguments

     ps(0, 1, 2)
     ps(a=1, c=2)

    Assess parameter values of this parameter set

     ps[0] == ps['a'] == ps.a

    A parameter set is a sequence:

     list(ps)

    Assess the parameter set properties as arrays
     [ps.maxbound, ps.minbound, ps.optguess, ps.step, ps.random]


    """

    def __init__(self, param_info):
        """
        Creates a set of parameters from a parameter info array.
        To create the parameter set from a setup use either:
         setup = ...
         ps = ParameterSet(get_parameters_array(setup))

        or you can just use a function for this:

         ps = create_set(setup)

        :param param_info: A record array containing the properties of the parameters
               of this set.
        """
        self.__lookup = dict(
            ("p" + x if x.isdigit() else x, i) for i, x in enumerate(param_info["name"])
        )
        self.__info = param_info

    def __call__(self, *values, **kwargs):
        """
        Populates the values ('random') of the parameter set with new data
        :param values: Contains the new values or omitted.
                       If given, the number of values needs to match the number
                       of parameters
        :param kwargs: Can be used to set only single parameter values
        :return:
        """
        if values:
            if len(self.__info) != len(values):
                raise ValueError(
                    "Given values do are not the same length as the parameter set"
                )
            self.__info["random"][:] = values
        for k in kwargs:
            try:
                self.__info["random"][self.__lookup[k]] = kwargs[k]
            except KeyError:
                raise TypeError("{} is not a parameter of this set".format(k))
        return self

    def __len__(self):
        return len(self.__info["random"])

    def __iter__(self):
        return iter(self.__info["random"])

    def __getitem__(self, item):
        """
        Provides item access
         ps[0] == ps['a']

        :raises: KeyError, IndexError and TypeError
        """
        if type(item) is str:
            item = self.__lookup[item]
        return self.__info["random"][item]

    def __setitem__(self, key, value):
        """
        Provides setting of item
         ps[0] = 1
         ps['a'] = 2
        """
        if key in self.__lookup:
            key = self.__lookup[key]
        self.__info["random"][key] = value

    def __getattr__(self, item):
        """
        Provides the attribute access like
         print(ps.a)
        """
        if item.startswith("_"):
            raise AttributeError(
                "{} is not a member of this parameter set".format(item)
            )
        elif item in self.__lookup:
            return self.__info["random"][self.__lookup[item]]
        elif item in self.__info.dtype.names:
            return self.__info[item]
        else:
            raise AttributeError(
                "{} is not a member of this parameter set".format(item)
            )

    def __setattr__(self, key, value):
        """
        Provides setting of attributes
         ps.a = 2
        """
        # Allow normal usage
        if key.startswith("_") or key not in self.__lookup:
            return object.__setattr__(self, key, value)
        else:
            self.__info["random"][self.__lookup[key]] = value

    def __str__(self):
        return "parameters({})".format(
            ", ".join(
                "{}={:g}".format(k, self.__info["random"][i])
                for i, k in enumerate(self.__info["name"])
            )
        )

    def __repr__(self):
        return "spotpy.parameter.ParameterSet()"

    def __dir__(self):
        """
        Helps to show the field names in an interactive environment like IPython.
        See: http://ipython.readthedocs.io/en/stable/config/integrating.html

        :return: List of method names and fields
        """
        attrs = [attr for attr in vars(type(self)) if not attr.startswith("_")]
        return attrs + list(self.__info["name"]) + list(self.__info.dtype.names)

    def set_by_array(self, array):
        for i, a in enumerate(array):
            self.__setitem__(i, a)

    def copy(self):
        return ParameterSet(copy.deepcopy(self.__info))


def get_classes():
    keys = []
    current_module = sys.modules[__name__]
    for key in dir(current_module):
        if isinstance(getattr(current_module, key), type):
            keys.append(key)
    return keys


def generate(parameters):
    """
    This function generates a parameter set from a list of parameter objects. The parameter set
    is given as a structured array in the format the parameters function of a setup expects
    :parameters: A sequence of parameter objects
    """
    dtype = [
        ("random", "<f8"),
        ("name", "|U100"),
        ("step", "<f8"),
        ("optguess", "<f8"),
        ("minbound", "<f8"),
        ("maxbound", "<f8"),
        ("as_int", "bool"),
    ]

    return np.fromiter(
        (param.astuple() for param in parameters), dtype=dtype, count=len(parameters)
    )


def check_parameter_types(parameters, unaccepted_parameter_types):
    if unaccepted_parameter_types:
        problems = []
        for param in parameters:
            for param_type in unaccepted_parameter_types:
                if type(param) == param_type:
                    problems.append([param, param_type])

        if problems:
            problem_string = ", ".join(
                "{} is {}".format(param, param_type) for param, param_type in problems
            )
            raise TypeError(
                "List parameters are only accepted for Monte Carlo (MC) sampler: "
                + problem_string
            )

    return parameters


def get_parameters_array(setup, unaccepted_parameter_types=()):
    """
    Returns the parameter array from the setup
    """
    # Put the parameter arrays as needed here, they will be merged at the end of this
    # function
    param_arrays = []
    # Get parameters defined with the setup class
    setup_parameters = get_parameters_from_setup(setup)
    check_parameter_types(setup_parameters, unaccepted_parameter_types)
    param_arrays.append(
        # generate creates the array as defined in the setup API
        generate(setup_parameters)
    )

    if hasattr(setup, "parameters") and callable(setup.parameters):
        # parameters is a function, as defined in the setup API up to at least spotpy version 1.3.13
        param_arrays.append(setup.parameters())

    # Return the class and the object parameters together
    res = np.concatenate(param_arrays)
    return res


def find_constant_parameters(parameter_array):
    """
    Checks which parameters are constant
    :param parameter_array: Return array from parameter.get_parameter_array(setup)
    :return: A True / False array with the len(result) == len(parameter_array)
    """
    return parameter_array["maxbound"] - parameter_array["minbound"] == 0.0


def create_set(setup, valuetype="random", **kwargs):
    """
    Returns a named tuple holding parameter values, to be used with the simulation method of a setup

    This function is typically used to test models, before they are used in a sampling

    Usage:
     import spotpy
     from spotpy.examples.spot_setup_rosenbrock import spot_setup
     model = spot_setup()
     param_set = spotpy.parameter.create_set(model, x=2)
     result = model.simulation(param_set)

    :param setup: A spotpy compatible Model object
    :param valuetype: Select between 'optguess' (defaul), 'random', 'minbound' and 'maxbound'.
    :param kwargs: Any keywords can be used to set certain parameters to fixed values
    :return: namedtuple of parameter values
    """

    # Get the array of parameter realizations
    params = get_parameters_array(setup)

    # Create the namedtuple from the parameter names
    partype = ParameterSet(params)

    # Return the namedtuple with fitting names
    return partype(*params[valuetype], **kwargs)


def get_constant_indices(setup, unaccepted_parameter_types=(Constant,)):
    """
    Returns a list of the class defined parameters, and
    overwrites the names of the parameters.
    By defining parameters as class members, as shown below,
    one can omit the parameters function of the setup.

    Usage:
     from spotpy import parameter
     class SpotpySetup:
         # Add parameters p1 & p2 to the setup.
         p1 = parameter.Uniform(20, 100)
         p2 = parameter.Gamma(2.3)

    setup = SpotpySetup()
    parameters = parameter.get_parameters_from_setup(setup)
    """

    # Get all class variables
    cls = type(setup)
    class_variables = vars(cls).items()

    par_index = []
    i = 0
    contained_class_parameter = False
    # for i in range(len(class_variables)):
    for attrname, attrobj in class_variables:
        # Check if it is an spotpy parameter
        if isinstance(attrobj, Base):
            contained_class_parameter = True
            if isinstance(attrobj, unaccepted_parameter_types):
                par_index.append(i)
            i += 1

    if contained_class_parameter:
        return par_index


def get_parameters_from_setup(setup):
    """
    Returns a list of the class defined parameters, and
    overwrites the names of the parameters.
    By defining parameters as class members, as shown below,
    one can omit the parameters function of the setup.

    Usage:

     from spotpy import parameter
     class SpotpySetup:
         # Add parameters p1 & p2 to the setup.
         p1 = parameter.Uniform(20, 100)
         p2 = parameter.Gamma(2.3)

     setup = SpotpySetup()
     parameters = parameter.get_parameters_from_setup(setup)
    """

    # Get all class variables
    cls = type(setup)
    class_variables = vars(cls).items()

    parameters = []
    for attrname, attrobj in class_variables:
        # Check if it is an spotpy parameter
        if isinstance(attrobj, Base):
            #            if isinstance(attrobj, unaccepted_parameter_types):
            #                pass
            #            # Set the attribute name
            #            else:
            if not attrobj.name:
                attrobj.name = attrname
            # Add parameter to dict
            parameters.append(attrobj)

    # Get the parameters list of setup if the parameter attribute is not a method:
    if hasattr(setup, "parameters") and not callable(setup.parameters):
        # parameters is not callable, assume a list of parameter objects.
        # Generate the parameters array from it and append it to our list
        parameters.extend(setup.parameters)

    return parameters
