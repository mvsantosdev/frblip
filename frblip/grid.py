import numpy

from scipy.interpolate import RectBivariateSpline

from astropy.coordinates import CartesianRepresentation
from astropy.coordinates.matrix_utilities import rotation_matrix


class CartesianGrid(object):

    def __init__(self, xrange, yrange, grid, az=None, alt=None,
                 rotation=None, **kwargs):

        if grid.ndim == 3:

            n_beams, xdim, ydim = grid.shape

            x = numpy.linspace(*xrange, xdim)
            y = numpy.linspace(*yrange, ydim)

            self.patterns = [
                RectBivariateSpline(y, x, g,
                                    kx=1, ky=1)
                for g in grid
            ]

            self.response = self._multiple_grids
            self.mesh = self._mesh_multiple_grids

        elif grid.ndim == 2:

            xdim, ydim = grid.shape

            x = numpy.linspace(*xrange, xdim)
            y = numpy.linspace(*yrange, ydim)

            self.pattern = RectBivariateSpline(y, x, grid,
                                               kx=1, ky=1)

            if (az is None) and (alt is None):

                self.response = self._unique_grid
                self.mesh = self._mesh_unique_grid

            else:

                self.response = self._shifted_unique_grid
                self.mesh = self._mesh_shifted_unique_grid

                self.rotations = []

                for iaz, ialt in zip(az, alt):

                    rot180z = rotation_matrix(180, axis='z')
                    rot90y = rotation_matrix(90, axis='y')

                    az_shift = rot180z @ rotation_matrix(iaz, axis='z')
                    alt_shift = rot90y @ rotation_matrix(-ialt, axis='y')

                    shift = az_shift @ alt_shift
                    shift = numpy.linalg.inv(shift)

                    self.rotations.append(shift)

    def __call__(self, azalt):

        xyz = azalt.represent_as('cartesian')

        return self.response(xyz)

    def _unique_grid(self, xyz):

        return self.pattern.ev(xyz.y.value,
                               xyz.x.value)

    def _multiple_grids(self, xyz):

        return numpy.column_stack([
                pattern.ev(xyz.y.value,
                           xyz.x.value)
                for pattern in self.patterns
            ])

    def _shifted_unique_grid(self, xyz):

        return numpy.column_stack([
                self._unique_grid(xyz.transform(rotation))
                for rotation in self.rotations
            ])

    def _mesh_unique_grid(self, x, y):

        return self.pattern(y, x)

    def _mesh_multiple_grids(self, x, y):

        return numpy.stack([
                pattern(y, x)
                for pattern in self.patterns
            ])

    def _mesh_shifted_unique_grid(self, x, y):

        X, Y = numpy.meshgrid(x, y)
        Z = numpy.sqrt(1 - X**2 - Y**2)

        xyz0 = CartesianRepresentation(x=X.ravel(),
                                       y=Y.ravel(),
                                       z=Z.ravel())

        responses = []

        for rotation in self.rotations:

            xyz = xyz0.transform(rotation)

            response = self.pattern.ev(xyz.y.value,
                                       xyz.x.value)
            response = response.reshape(Z.shape)

            responses.append(response)

        return numpy.stack(responses)
