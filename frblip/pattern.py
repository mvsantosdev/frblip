import numpy

from scipy.special import j1

from astropy import units

from .utils import _all_sky_area, angular_separation


class FunctionalPattern(object):

    def __init__(self, kind, az, alt,
                 maximum_gain, **kwargs):

        self.az = az * units.degree
        self.alt = alt * units.degree

        solid_angle = 10**(- 0.1 * maximum_gain)
        solid_angle = _all_sky_area * solid_angle
        solid_angle = solid_angle.to(units.sr).value

        arg = 1 - solid_angle / (2 * numpy.pi)

        radius = numpy.arccos(arg) * units.rad

        self.radius = radius.to(units.degree)

        self.response = self.__getattribute__(kind)

    def __call__(self, azalt):

        az = azalt.az
        alt = azalt.alt

        arcs = angular_separation(az.reshape(-1, 1),
                                  alt.reshape(-1, 1),
                                  self.az, self.alt)

        rescaled_arc = (arcs / self.radius).to(1).value

        return self.response(rescaled_arc)

    def tophat(self, x):

        return (numpy.abs(x) <= 1).astype(numpy.float)

    def gaussian(self, x):

        return numpy.exp(-x**2)

    def bessel(self, x):

        return (j1(2 * x) / x)**2
