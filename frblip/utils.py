import os

import numpy

from functools import partial

from collections import namedtuple

from itertools import repeat, cycle, product, combinations

from astropy import coordinates, units

from scipy.stats import rvs_ratio_uniforms

from scipy.integrate import cumtrapz

_ROOT = os.path.abspath(os.path.dirname(__file__))
_DATA = os.path.join(_ROOT, 'data')


def paired_shapes(shape):

    nscopes = shape.size

    if nscopes < 2:
        return numpy.atleast_2d(shape)

    diag_shape = numpy.diag(shape - 1).astype(numpy.int32)
    self_shapes = numpy.ones_like(diag_shape, dtype=numpy.int32) + diag_shape

    j1, j2 = numpy.column_stack([*combinations(range(nscopes), 2)])
    i = numpy.arange(j1.size)

    cross_shapes = numpy.ones((j1.size, nscopes), dtype=numpy.int32)

    cross_shapes[i, j1] = shape[j1]
    cross_shapes[i, j2] = shape[j2]

    return numpy.row_stack((self_shapes, cross_shapes))


def xfactors(*n):

    N = numpy.array(n)
    factor = N * N.reshape(-1, 1)
    factor = factor - N * numpy.eye(N.size, dtype=numpy.int8)
    diag_idx = numpy.diag_indices_from(factor)
    tril_idx = numpy.triu_indices_from(factor, k=1)
    diag = factor[diag_idx] // 2
    triu = factor[tril_idx]

    return numpy.concatenate((diag, triu))


def sub_dict(kwargs, keys=None, flag='', pop=False,
             replace_flag='', apply=lambda x: x):

    dict_func = (lambda x: kwargs.pop(x, None)) if pop else kwargs.get

    def func(x):
        return apply(dict_func(x))

    flag_len = len(flag)
    keys = [*kwargs.keys()] if keys is None else keys
    keys = [key for key in keys if flag == key[:flag_len]]

    output = {
        key.replace(flag, replace_flag): func(key)
        for key in keys
    }

    return {k: v for k, v in output.items() if v is not None}


def load_params(input_dict):

    output_dict = input_dict.copy()

    input_units = sub_dict(input_dict, flag='u_', pop=True,
                           apply=units.Unit)

    output_dict.update({
        key: input_dict[key] * units.Unit(input_units[key])
        for key in input_units
    })

    return output_dict


def load_file(file):

    output = numpy.load(file, allow_pickle=True)
    output = dict(output)

    output.update({
            key: value.item()
            for key, value in output.items()
            if value.ndim == 0
        })

    return output


def simps(y):

    m = y[..., 1::2]
    a = y[..., :-1:2]
    b = y[..., 2::2]

    return (a + b + 4*m) / 6
