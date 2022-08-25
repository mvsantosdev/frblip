import numpy

from astropy import units

from scipy.integrate import cumtrapz
from scipy.stats import rv_continuous


class Redshift(rv_continuous):
    """ """

    def __init__(self, zmin=0.0, zmax=6, cosmology=None, eps=1e-3):

        super().__init__()

        self.zmin = zmin
        self.zmax = zmax
        self.cosmology = cosmology

        self.pdf_norm = 1.0 / units.Mpc**3

        ngrid = 1 + int((zmax - zmin) // eps)
        self.zgrid = numpy.linspace(zmin, zmax, ngrid)

        pdf = self.pdf(self.zgrid)

        self.cdf_grid = cumtrapz(x=self.zgrid, y=pdf, initial=0.0)
        self.pdf_norm = self.pdf_norm / self.cdf_grid[-1]
        self.cdf_grid = self.cdf_grid / self.cdf_grid[-1]
        self.isf_grid = 1 - self.cdf_grid

    def _get_support(self):
        """ """

        return self.zmin, self.zmax

    def angular_density(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        diff_co_vol = self.cosmology.differential_comoving_volume(z)
        return diff_co_vol / (1 + z)

    def density(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        angular_density = self.angular_density(z)
        return units.spat * angular_density

    def _pdf(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        density = self.density(z)
        return (self.pdf_norm * density).to(1).value

    def _logpdf(self, x):
        """

        Parameters
        ----------
        x :


        Returns
        -------

        """
        return numpy.log(self._pdf(x))

    def _cdf(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        return numpy.interp(x=z, xp=self.zgrid, fp=self.cdf_grid)

    def _logcdf(self, x):
        """

        Parameters
        ----------
        x :


        Returns
        -------

        """
        return numpy.log(self._cdf(x))

    def _sf(self, x):
        """

        Parameters
        ----------
        x :


        Returns
        -------

        """
        return 1 - self._cdf(x)

    def _logsf(self, x):
        """

        Parameters
        ----------
        x :


        Returns
        -------

        """
        return numpy.log(self._sf(x))

    def _ppf(self, u):
        """

        Parameters
        ----------
        u :


        Returns
        -------

        """
        return numpy.interp(x=u, xp=self.cdf_grid, fp=self.zgrid)

    def _isf(self, u):
        """

        Parameters
        ----------
        u :


        Returns
        -------

        """
        return numpy.interp(x=u, xp=self.isf_grid, fp=self.zgrid)
