import numpy

from scipy.special import j1

from astropy import units, coordinates

from .utils import _all_sky_area


class FunctionalPattern(object):

    def __init__(self, directivity, alt=90.0, az=0.0, kind='gaussian'):

        self.n_beam = numpy.size(az)

        altaz = coordinates.AltAz(alt=alt, az=az)

        self.Offsets = coordinates.SkyOffsetFrame(origin=altaz)

        solid_angle = 4 * numpy.pi / directivity.to(1 / units.sr)
        arg = 1 - solid_angle / (2 * numpy.pi * units.sr)

        self.radius = numpy.arccos(arg)

        self.response = self.__getattribute__(kind)

    def __call__(self, AltAz):

        AltAzOffs = [
            AltAz.transform_to(self.Offsets[i])
            for i in range(self.n_beam)
        ]

        cossines = numpy.column_stack([
            AltAzOff.cartesian.x
            for AltAzOff in AltAzOffs
        ])

        arcs = numpy.arccos(cossines)

        rescaled_arc = (arcs / self.radius).to(1).value

        return self.response(rescaled_arc)

    def tophat(self, x):

        return (numpy.abs(x) <= 1).astype(numpy.float)

    def gaussian(self, x):

        return numpy.exp(-x**2)

    def bessel(self, x):

        return (j1(2 * x) / x)**2
