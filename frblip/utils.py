import os

import numpy

from collections import namedtuple

from itertools import repeat, cycle, product

from astropy import coordinates, units

from scipy.stats import rvs_ratio_uniforms

from scipy.integrate import cumtrapz


_all_sky_area = 4 * numpy.pi * units.sr

_ROOT = os.path.abspath(os.path.dirname(__file__))
_DATA = os.path.join(_ROOT, 'data')


def load(file):

    output = numpy.load(file, allow_pickle=True)
    output = dict(output)

    output.update({
            key: value.item()
            for key, value in output.items()
            if value.ndim == 0
        })

    return output


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


def load_params(file):

    input_dict = load(file)

    input_units = sub_dict(input_dict, flag='u_', pop=True,
                           apply=units.Unit)

    input_dict.update({
        key: input_dict[key] * units.Unit(input_units[key])
        for key in input_units
    })

    return input_dict


def schechter(x, alpha):

    return (x**alpha) * numpy.exp(-x)


def simps(y):

    m = y[..., 1::2]
    a = y[..., :-1:2]
    b = y[..., 2::2]

    return (a + b + 4*m) / 6


def rvs_from_pdf(pdf, xmin, xmax, size=1):

    xbounds = np.array([xmin, xmax])
    fbounds = pdf(xbounds)

    vbound = xbounds * np.sqrt(fbounds)

    vmin, vmax = np.sort(vbound)

    umax = np.sqrt(fbound.max())

    return rvs_ratio_uniforms(pdf, umax=umax, vmin=vmin, vmax=vmax, size=size)


def rvs_from_cdf(pdf, xmin, xmax, precision=1e-5, size=1, **kwargs):

    L = xmax - xmin
    d = precision * L

    N = int(L // d)

    x = numpy.linspace(xmin, xmax, N)

    y = pdf(x, **kwargs)

    cdf = cumtrapz(x=x, y=y, initial=0)
    cdf = cdf / cdf[-1]

    U = numpy.random.random(size)

    return numpy.interp(x=U, xp=cdf, fp=x)
