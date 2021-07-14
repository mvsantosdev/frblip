import numpy

from sparse import COO

from scipy.special import j1

from astropy import units, coordinates


class FunctionalPattern(object):

    def __init__(self, directivity, alt=90.0, az=0.0, kind='gaussian'):

        self.n_beam = numpy.size(az)
        self.response = self.__getattribute__(kind)
        self.set_radius(directivity)
        self.set_offset(alt, az)

    def set_radius(self, directivity):

        solid_angle = 4 * numpy.pi / directivity.to(1 / units.sr)
        arg = 1 - solid_angle / (2 * numpy.pi * units.sr)
        self.radius = numpy.arccos(arg)

    def set_offset(self, alt, az):

        altaz = coordinates.AltAz(alt=alt, az=az)
        self.offsets = coordinates.SkyOffsetFrame(origin=altaz)
        self.n_beam = az.size

    def __call__(self, AltAz):

        AltAzOffs = [
            AltAz.transform_to(self.offsets[i])
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

        return numpy.abs(x) <= 1

    def gaussian(self, x):

        return numpy.exp(-x**2)

    def bessel(self, x):

        return numpy.nan_to_num(j1(2 * x) / x, nan=1.0)**2
