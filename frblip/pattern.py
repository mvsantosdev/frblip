import numpy
from scipy.special import j1
from astropy import coordinates

from functools import cached_property


class FunctionalPattern(object):
    """ """

    def __init__(self, radius, alt=90.0, az=0.0, kind='gaussian'):

        self.radius = radius
        self.alt, self.az = alt, az

        self.beams = numpy.unique([self.alt.size, self.az.size])
        assert self.beams.size == 1
        self.beams = self.beams.item()
        if (self.radius.size > 1) and (self.beams == 1):
            self.beams = self.radius.size

        if (self.beams > 1) and (self.radius.size == 1):
            self.radius = numpy.tile(self.radius, self.beams)

        key = '_FunctionalPattern__{}'.format(kind)
        self.response = self.__getattribute__(key)

    @cached_property
    def offsets(self):
        altaz = coordinates.AltAz(alt=self.alt, az=self.az)
        return coordinates.SkyOffsetFrame(origin=altaz)

    def __call__(self, altaz):

        altazoffs = [
            altaz.transform_to(self.offsets[i])
            for i in range(self.beams)
        ]

        cossines = numpy.column_stack([
            altazoff.cartesian.x
            for altazoff in altazoffs
        ])

        arcs = numpy.arccos(cossines)
        rescaled_arc = (arcs / self.radius).to(1).value
        return self.response(rescaled_arc)

    def __tophat(self, x):
        return numpy.abs(x) <= 1

    def __gaussian(self, x):
        return numpy.exp(-x**2)

    def __bessel(self, x):
        return numpy.nan_to_num(j1(2 * x) / x, nan=1.0)**2
