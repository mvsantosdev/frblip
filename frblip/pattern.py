import numpy

from scipy.special import j1

from astropy import units, coordinates

from .utils import _all_sky_area


class FunctionalPattern(object):

    def __init__(self, maximum_gain, alt=90.0, az=0.0,
                 kind='gaussian', **kwargs):

        self.n_beam = numpy.size(az)

        AltAz = coordinates.AltAz(alt=alt * units.degree,
                                  az=az * units.degree)

        self.Offsets = coordinates.SkyOffsetFrame(origin=AltAz)

        solid_angle = 10**(- 0.1 * maximum_gain)
        solid_angle = _all_sky_area * solid_angle
        solid_angle = solid_angle.to(units.sr).value

        arg = 1 - solid_angle / (2 * numpy.pi)

        self.radius = numpy.arccos(arg) * units.rad

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

        rescaled_arc = arcs / self.radius

        return self.response(rescaled_arc).to(1).value

    def tophat(self, x):

        return (numpy.abs(x) <= 1).astype(numpy.float)

    def gaussian(self, x):

        return numpy.exp(-x**2)

    def bessel(self, x):

        return (j1(2 * x) / x)**2
