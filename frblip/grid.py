import numpy

from astropy import units, coordinates

from scipy.interpolate import RectBivariateSpline


class CartesianGrid(object):

    def __init__(self, grid, xrange, yrange, alt=90, az=0.0,
                 **kwargs):

        self._min = grid.min()

        if grid.ndim == 3:

            self.n_beam, xdim, ydim = grid.shape

            x = numpy.linspace(*xrange, xdim)
            y = numpy.linspace(*yrange, ydim)

            self.patterns = [
                RectBivariateSpline(y, x, g,
                                    kx=3, ky=3)
                for g in grid
            ]

            self.response = self._multiple_grids

        elif grid.ndim == 2:

            xdim, ydim = grid.shape

            x = numpy.linspace(*xrange, xdim)
            y = numpy.linspace(*yrange, ydim)

            self.pattern = RectBivariateSpline(y, x, grid,
                                               kx=3, ky=3)

            self.response = self._unique_grid

            self.n_beam = numpy.size(alt)

            AltAz = coordinates.AltAz(alt=alt * units.degree,
                                      az=az * units.degree)

            self.Offsets = coordinates.SkyOffsetFrame(origin=AltAz)

    def __call__(self, AltAz):

        return self.response(AltAz).clip(self._min)

    def _unique_grid(self, AltAz):

        AltAzOffs = [
            AltAz.transform_to(self.Offsets[i])
            for i in range(self.n_beam)
        ]

        x = numpy.column_stack([
            AltAzOff.cartesian.y
            for AltAzOff in AltAzOffs
        ])

        y = numpy.column_stack([
            AltAzOff.cartesian.z
            for AltAzOff in AltAzOffs
        ])

        return self.pattern.ev(y, x)

    def _multiple_grids(self, AltAz):

        x = AltAz.cartesian.x
        y = AltAz.cartesian.y

        return numpy.column_stack([
                pattern.ev(y, x)
                for pattern in self.patterns
            ])
