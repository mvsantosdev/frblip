import numpy

from scipy.stats import rv_continuous
from scipy.integrate import quad, cumtrapz


class Schechter(rv_continuous):

    def __init__(self, xmin, alpha, eps=1e-3):

        super().__init__()
        self.xmin = xmin
        self.xmax = 1 / eps
        self.alpha = alpha

        volume, err = quad(self.schechter, self.xmin, self.xmax)
        self.pdf_norm = 1 / volume

        dx = self.xmax - self.xmin
        ngrid = 1 + int(dx // eps)
        log_xmin = numpy.log10(self.xmin)
        log_xmax = numpy.log10(self.xmax)

        self.xgrid = numpy.logspace(log_xmin, log_xmax, ngrid)

        pdf = self.pdf(self.xgrid)
        self.cdf_grid = cumtrapz(x=self.xgrid, y=pdf, initial=0.0)
        self.isf_grid = 1 - self.cdf_grid

    def _get_support(self):
        return self.xmin, self.xmax

    def schechter(self, x):
        return x**self.alpha * numpy.exp(-x)

    def _pdf(self, x):
        return self.pdf_norm * self.schechter(x)

    def _logpdf(self, x):
        return numpy.log(self._pdf(x))

    def _cdf(self, x):
        return numpy.interp(x=x, xp=self.xgrid, fp=self.cdf_grid)

    def _logcdf(self, x):
        return numpy.log(self._cdf(x))

    def _sf(self, x):
        return 1 - self._cdf(x)

    def _logsf(self, x):
        return numpy.log(self._sf(x))

    def _ppf(self, u):
        return numpy.interp(x=u, xp=self.cdf_grid, fp=self.xgrid)

    def _isf(self, u):
        return numpy.interp(x=u, xp=self.isf_grid, fp=self.xgrid)

    def log_rvs(self, size):
        sample = self.rvs(size=size)
        return numpy.log10(sample)
