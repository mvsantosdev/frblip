import numpy

from astropy import units, coordinates

from scipy.interpolate import RectBivariateSpline


class CartesianGrid(object):

    def __init__(self, grid, xrange, yrange, alt=90, az=0.0):

        self.__min = grid.min()

        if grid.ndim == 3:

            self.n_beam, xdim, ydim = grid.shape

            x = numpy.linspace(*xrange, xdim)
            y = numpy.linspace(*yrange, ydim)

            self.patterns = [
                RectBivariateSpline(y, x, g, kx=3, ky=3)
                for g in grid
            ]

            self.response = self.__multiple_grids

        elif grid.ndim == 2:

            xdim, ydim = grid.shape

            x = numpy.linspace(*xrange, xdim)
            y = numpy.linspace(*yrange, ydim)
            z = numpy.sqrt(1 - x**2 - y**2) * grid

            self.pattern = RectBivariateSpline(y, x, z, kx=3, ky=3)

            self.response = self.__unique_grid

            self.n_beam = numpy.size(alt)

            alt = alt - 90.0 * units.deg
            altaz = coordinates.AltAz(alt=alt, az=az)

            self.offsets = coordinates.SkyOffsetFrame(origin=altaz)

    def __call__(self, altaz):

        return self.response(altaz).clip(self.__min)

    def __unique_grid(self, altaz):

        altazoffs = [
            altaz.transform_to(self.offsets[i])
            for i in range(self.n_beam)
        ]

        x = numpy.column_stack([
            altazoff.cartesian.x
            for altazoff in altazoffs
        ])

        y = numpy.column_stack([
            altazoff.cartesian.y
            for altazoff in altazoffs
        ])

        return self.pattern.ev(y, x)

    def __multiple_grids(self, altaz):

        x = altaz.cartesian.x
        y = altaz.cartesian.y

        return numpy.column_stack([
                pattern.ev(y, x)
                for pattern in self.patterns
            ])

    def set_directions(self, alt, az):

        altaz = coordinates.AltAz(alt=alt, az=az)
        self.offsets = coordinates.SkyOffsetFrame(origin=altaz)
        self.n_beam = az.size
