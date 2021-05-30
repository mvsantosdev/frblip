import numpy

from astropy import units
from astropy import cosmology

from scipy.stats import rv_continuous
from scipy.integrate import cumtrapz


class Redshift(rv_continuous):

    def __init__(self, zmin=0.0, zmax=6, eps=1e-3,
                 cosmology=cosmology.Planck18_arXiv_v2):

        super().__init__()
        self.cosmology = cosmology

        self.zmin = zmin
        self.zmax = zmax

        self.pdf_norm = 1.0 / units.Mpc**3

        ngrid = 1 + int((zmax - zmin) // eps)
        self.zgrid = numpy.linspace(zmin, zmax, ngrid)

        pdf = self.pdf(self.zgrid)

        self.cdf_grid = cumtrapz(x=self.zgrid, y=pdf, initial=0.0)
        self.pdf_norm = self.pdf_norm / self.cdf_grid[-1]
        self.cdf_grid = self.cdf_grid / self.cdf_grid[-1]
        self.isf_grid = 1 - self.cdf_grid

    def _get_support(self):
        return self.zmin, self.zmax

    def density(self, z):
        diff_co_vol = self.cosmology.differential_comoving_volume(z)
        density = units.astrophys.sp * diff_co_vol
        return density / (1 + z)

    def _pdf(self, z):
        density = self.density(z)
        return (self.pdf_norm * density).to(1).value

    def _logpdf(self, x):
        return numpy.log(self._pdf(x))

    def _cdf(self, z):
        return numpy.interp(x=t, xp=self.zgrid, fp=self.cdf_grid)

    def _logcdf(self, x):
        return numpy.log(self._cdf(x))

    def _sf(self, x):
        return 1 - self._cdf(x)

    def _logsf(self, x):
        return numpy.log(self._sf(x))

    def _ppf(self, u):
        return numpy.interp(x=u, xp=self.cdf_grid, fp=self.zgrid)

    def _isf(self, u):
        return numpy.interp(x=u, xp=self.isf_grid, fp=self.zgrid)