import numpy

from scipy.stats import norm
from scipy.stats import rv_continuous


class Mixture(rv_continuous):

    def __init__(self, loc, scale, weights):

        super().__init__()

        self.loc = loc
        self.scale = scale
        self.weight = weights / weights.sum()

    def _get_support(self):
        return - numpy.inf, numpy.inf

    def _pdf(self, x):
        pdfs = numpy.stack([
            norm.pdf(x=x, loc=loc, scale=scale)
            for loc, scale in zip(self.loc, self.scale)
        ], axis=-1)

        return (pdfs * self.weight).sum(-1)

    def _logpdf(self, x):
        return numpy.log(self._pdf(x))

    def _cdf(self, x):
        cdfs = numpy.stack([
            norm.cdf(x=x, loc=loc, scale=scale)
            for loc, scale in zip(self.loc, self.scale)
        ], axis=-1)

        return (cdfs * self.weight).sum(-1)

    def _logcdf(self, x):
        return numpy.log(self._cdf(x))

    def _sf(self, x):
        return 1 - self._cdf(x)

    def _logsf(self, x):
        return numpy.log(self._sf(x))

    def _rvs(self, size=None, random_state=None):
        idxs = numpy.random.choice(self.loc.size, size=size, p=self.weight)
        return numpy.random.normal(loc=self.loc[idxs], scale=self.scale[idxs])
