import numpy

from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import rv_continuous


class Mixture(rv_continuous):
    """ """

    def __init__(self, loc, scale, weights, trunc=False):

        super().__init__()

        self.loc = loc
        self.scale = scale
        self.weight = weights / weights.sum()

        self.xmin = 0 if trunc else - numpy.inf
        self._rvs = self._trunc if trunc else self._no_trunc

    def _get_support(self):
        """ """
        return self.xmin, numpy.inf

    def _pdf(self, x):
        """

        Parameters
        ----------
        x :


        Returns
        -------

        """
        pdfs = numpy.stack([
            norm.pdf(x=x, loc=loc, scale=scale)
            for loc, scale in zip(self.loc, self.scale)
        ], axis=-1)

        return (pdfs * self.weight).sum(-1)

    def _logpdf(self, x):
        """

        Parameters
        ----------
        x :


        Returns
        -------

        """
        return numpy.log(self._pdf(x))

    def _cdf(self, x):
        """

        Parameters
        ----------
        x :


        Returns
        -------

        """
        cdfs = numpy.stack([
            norm.cdf(x=x, loc=loc, scale=scale)
            for loc, scale in zip(self.loc, self.scale)
        ], axis=-1)

        return (cdfs * self.weight).sum(-1)

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

    def _no_trunc(self, size=None, random_state=None):
        """

        Parameters
        ----------
        size :
             (Default value = None)
        random_state :
             (Default value = None)

        Returns
        -------

        """
        idxs = numpy.random.choice(self.loc.size, size=size, p=self.weight)
        return numpy.random.normal(loc=self.loc[idxs], scale=self.scale[idxs])

    def _trunc(self, size=None, random_state=None):
        """

        Parameters
        ----------
        size :
             (Default value = None)
        random_state :
             (Default value = None)

        Returns
        -------

        """
        idxs = numpy.random.choice(self.loc.size, size=size, p=self.weight)
        loc, scale = self.loc[idxs], self.scale[idxs]
        z = truncnorm.rvs(a=-loc/scale, b=numpy.inf, size=size)
        return scale * z + loc
