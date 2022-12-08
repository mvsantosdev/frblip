import os

import numpy
import pandas
from astropy import units

from ..mixture import Mixture

from ...cosmology import Cosmology

_ROOT = os.path.abspath(os.path.dirname(__file__))
_DATA = _ROOT.replace('random/dispersion_measure', 'data')

unit = units.pc / units.cm**3
galactic_edge = 30 * units.kpc


class HostGalaxyDM():
    """ """

    def __init__(self, source='luo18', model=('ALG', 'YMW16'),
                 cosmology='Planck_18', dist='lognormal'):

        path = '{}/{}_host.csv'.format(_DATA, source)

        models = pandas.read_csv(path, index_col=[0, 1], header=[0, 1])

        loc = models.loc['loc', model].values.ravel()
        scale = models.loc['scale', model].values.ravel()
        weight = models.loc['weight', model].values.ravel()

        self.mixture = Mixture(loc, scale, weight, dist == 'normal')
        if isinstance(cosmology, str):
            self.cosmology = Cosmology(cosmology)
        elif isinstance(cosmology, Cosmology):
            self.cosmology = cosmology

        key = '_{}'.format(dist)
        self._dist = getattr(self, key)

    def _lognormal(self, z):
        logDM0 = self.mixture.rvs(size=z.shape)
        sfr = self.cosmology.star_formation_rate(z)
        sfr_ratio = numpy.sqrt(sfr / self.cosmology.sfr0)

        return sfr_ratio * (10**logDM0) * unit

    def _normal(self, z):
        DM0 = self.mixture.rvs(size=z.shape)
        sfr = self.cosmology.star_formation_rate(z)
        sfr_ratio = numpy.sqrt(sfr / self.cosmology.sfr0)

        return sfr_ratio * DM0 * unit

    def __call__(self, z):
        return self._dist(z)
