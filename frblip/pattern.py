import numpy
from scipy.special import j1
from astropy import coordinates


class FunctionalPattern(object):
    """ """

    def __init__(self, radius, alt=90.0, az=0.0, kind='gaussian'):

        self.beams = numpy.size(az)
        self.radius = radius
        self.set_directions(alt, az)
        key = '_FunctionalPattern__{}'.format(kind)
        self.response = self.__getattribute__(key)

    def set_directions(self, alt, az):
        """

        Parameters
        ----------
        alt :

        az :


        Returns
        -------

        """

        altaz = coordinates.AltAz(alt=alt, az=az)
        self.offsets = coordinates.SkyOffsetFrame(origin=altaz)
        self.beams = az.size

    def __call__(self, AltAz):

        AltAzOffs = [
            AltAz.transform_to(self.offsets[i])
            for i in range(self.beams)
        ]

        cossines = numpy.column_stack([
            AltAzOff.cartesian.x
            for AltAzOff in AltAzOffs
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
