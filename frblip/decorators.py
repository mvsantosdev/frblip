import os
import json
import bz2
import dill

import numpy
import xarray
from sparse import COO
from toolz.dicttoolz import valfilter

import inspect
from glob import glob
from functools import wraps

from astropy.units import Unit, Quantity


def fixargs(function, *args, **kwargs):

    varnames = function.__code__.co_varnames
    kw = dict(zip(varnames, args))

    keys = [*kwargs.keys()]
    new_keys = [*kw.keys()]
    intersec = numpy.intersect1d(keys, new_keys)

    if len(intersec) > 0:
        name = function.__name__
        val = intersec[0]
        raise TypeError(f"{name}() got multiple values for argument '{val}'")

    kwargs.update(kw)

    signature = inspect.signature(function)
    parameters = signature.parameters

    defaults = {
        val.name: val.default
        for val in parameters.values()
        if val.default is not inspect._empty
        and val.name not in kwargs
    }

    kwargs.update(defaults)

    return kwargs


def default_units(
    output_unit: str | Unit | Quantity | None = None,
    **units
):

    def inner_decorator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):

            kwargs = fixargs(function, *args, **kwargs)

            for name, unit in units.items():

                un = Unit(unit)

                value = kwargs[name]
                if value is None:
                    kwargs[name] = value
                elif isinstance(value, Quantity):
                    kwargs[name] = value.to(un)
                elif isinstance(value, str):
                    value = Quantity(value)
                    kwargs[name] = value.to(un)
                else:
                    kwargs[name] = value * un

            source = kwargs.pop('self')
            output = function(source, **kwargs)

            if output_unit is None:
                return output
            elif isinstance(output_unit, (str, Unit)):

                un = Unit(output_unit)

                if isinstance(output, Quantity):
                    return output.to(un)
                elif isinstance(output, (numpy.ndarray, float, int)):
                    return output * un
            else:
                raise TypeError(f'{output_unit} is not a valid astropy unit.')

        return wrapper

    return inner_decorator


def xarrayfy(**dimensions):

    def inner_decorator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):

            kwargs = fixargs(function, *args, **kwargs)

            for name, dims in dimensions.items():
                value = kwargs[name]
                if isinstance(value, xarray.DataArray):
                    assert value.dims == dims, f'{name}.dims is not {dims}'
                elif hasattr(value, '__iter__'):
                    kwargs[name] = xarray.DataArray(
                        value, dims=dims
                    )
                else:
                    raise TypeError(f'{name} is not an iterable')
            return function(**kwargs)
        return wrapper
    return inner_decorator


def todense_option(method):
    @wraps(method)
    def wrapper(*args, **kwargs):

        kwargs = fixargs(method, *args, **kwargs)
        output = method(**kwargs)

        todense = kwargs.get('todense', False)

        if todense:
            data = getattr(output, 'data', None)
            if isinstance(data, COO):
                return output.as_numpy()
        return output

    return wrapper


def observation_method(method):

    @wraps(method)
    def wrapper(*args, **kwargs):

        sampler = args[0]
        names = args[1:]

        if names == ():
            observations = sampler.observations
        else:
            observations = {
                name: sampler.observations[name]
                for name in names
            }

        if len(observations) == 1:
            [(_, observation)] = observations.items()
            return method(sampler, observation, **kwargs)

        observation = {
            name: method(sampler, observation, **kwargs)
            for name, observation in observations.items()
        }

        return valfilter(lambda x: x is not None, observation)

    return wrapper


def from_source(default_folder='', default_dict=None):

    def inner_decorator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):

            kwargs = fixargs(function, *args, **kwargs)
            source = kwargs.get('source')

            if isinstance(source, dict):

                input_dict = source

            elif isinstance(default_dict, dict) and (source in default_dict):

                input_dict = default_dict.get(source)

            else:

                if source is None:
                    return function(**kwargs)

                folder, file_name = os.path.split(source)
                _, ext = os.path.splitext(file_name)

                if (folder, ext) == ('', ''):
                    pattern = '{}/{}*'.format(default_folder, file_name)
                    paths = glob(pattern)
                    if len(paths) != 1:
                        raise FileNotFoundError(pattern)
                    [source] = paths

                    _, ext = os.path.splitext(source)

                if ext == '.pkl':
                    file = bz2.BZ2File(source, 'rb')
                    input_dict = dill.load(file)
                    file.close()
                elif ext == '.json':
                    file = open(source, 'r')
                    input_dict = json.load(file)
                    file.close()

            kwargs.update(input_dict)

            return function(**kwargs)

        return wrapper

    return inner_decorator
