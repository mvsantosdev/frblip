import os

import numpy

from astropy import coordinates, units

from scipy.stats import rvs_ratio_uniforms

from scipy.integrate import cumtrapz


_all_sky_area = 4 * numpy.pi * units.sr

_ROOT = os.path.abspath(os.path.dirname(__file__))
_DATA = os.path.join(_ROOT, 'data')


def azalt2uvw(az, alt):

    calt = numpy.cos(alt)

    u = - calt * numpy.sin(az)
    v = calt * numpy.cos(az)
    w = numpy.sin(alt)

    return u, v, w


def schechter(x, alpha):

    return (x**alpha) * numpy.exp(-x)


def simps(y):

    m = y[..., 1::2]
    a = y[..., :-1:2]
    b = y[..., 2::2]

    return (a + b + 4*m) / 6


def angular_separation(lon1, lat1, lon2, lat2):

    s1, s2 = numpy.sin(lat1), numpy.sin(lat2)
    c1, c2 = numpy.cos(lat1), numpy.cos(lat2)

    cosu = s1 * s2 + c1 * c2 * numpy.cos(lon1 - lon2)

    return numpy.arccos(cosu).to(units.degree)


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
