import os

import numpy

from astropy import coordinates, units


_ROOT = os.path.abspath(os.path.dirname(__file__))
_DATA = os.path.join(_ROOT, 'data')

def phi(x, alpha):

    return (x**alpha) * numpy.exp(-x)


def angular_separation(lon1, lat1, lon2, lat2):

    s1, s2 = numpy.sin(lat1), numpy.sin(lat2)
    c1, c2 = numpy.cos(lat1), numpy.cos(lat2)

    cosu = s1 * s2 + c1 * c2 * numpy.cos(lon1 - lon2)

    return numpy.arccos(cosu).to(units.degree)
