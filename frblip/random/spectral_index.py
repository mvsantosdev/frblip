import os

import numpy
import pandas
from astropy import units

from .mixture import Mixture

_ROOT = os.path.abspath(os.path.dirname(__file__))
_DATA = _ROOT.replace('random', 'data')

unit = units.pc / units.cm**3
galactic_edge = 30 * units.kpc


class SpectralIndex():
    """ """

    def __init__(self, params, source='spec_idx'):

        if isinstance(params, tuple):
            self.si_min, self.si_max = params
            self.rvs = self.__uniform
        elif isinstance(params, (float, int)):
            self.spectral_index = params
            self.rvs = self.__constant
        elif isinstance(params, str):
            path = '{}/{}.csv'.format(_DATA, source)
            df = pandas.read_csv(path, index_col=[0, 1])
            loc = df.loc['loc', params].values
            scale = df.loc['scale', params].values
            weight = df.loc['weight', params].values
            self.mixture = Mixture(loc, scale, weight)
            self.rvs = self.__mixture

    def __uniform(self, size):
        return numpy.random.uniform(self.si_min, self.si_max,
                                    size=size)

    def __constant(self, size):
        return numpy.full(size, self.spectral_index)

    def __mixture(self, size):
        return self.mixture.rvs(size=size)
