import os
import json
import bz2
import dill

import numpy
import xarray
import sparse

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


def default_units(**units):

    def inner_decorator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):

            kwargs = fixargs(function, *args, **kwargs)

            for name, unit in units.items():

                value = kwargs[name]
                if hasattr(value, 'unit'):
                    kwargs[name] = value.to(unit)
                elif isinstance(value, str):
                    value = Quantity(value)
                    kwargs[name] = value.to(unit)
                else:
                    if isinstance(unit, str):
                        kwargs[name] = value * Unit(unit)
                    else:
                        kwargs[name] = value * unit

            source = kwargs.pop('self')
            return function(source, **kwargs)

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


def todense_option(default=True):

    def inner_decorator(method):
        @wraps(method)
        def wrapper(*args, todense: bool = default, **kwargs):

            sampler = args[0]
            output = method(sampler, *args[1:], **kwargs)
            if todense:
                data = getattr(output, 'data', None)
                if isinstance(data, sparse.COO):
                    return output.as_numpy()
            return output

        return wrapper
    return inner_decorator


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

        return {
            name: method(sampler, observation, **kwargs)
            for name, observation in observations.items()
        }

    return wrapper


def from_file(default_folder):

    def inner_decorator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):

            kwargs = fixargs(function, *args, **kwargs)
            name = kwargs.get('name')

            if name is None:
                return function(**kwargs)

            folder, file_name = os.path.split(name)
            _, ext = os.path.splitext(file_name)

            if (folder, ext) == ('', ''):
                pattern = '{}/{}*'.format(default_folder, file_name)
                paths = glob(pattern)
                if len(paths) != 1:
                    raise FileNotFoundError(pattern)
                [name] = paths

                _, ext = os.path.splitext(name)

            if ext == '.pkl':
                file = bz2.BZ2File(name, 'rb')
                input_dict = dill.load(file)
                file.close()
            elif ext == '.json':
                file = open(name, 'r')
                input_dict = json.load(file)
                file.close()

            kwargs.update(input_dict)

            return function(**kwargs)

        return wrapper

    return inner_decorator
