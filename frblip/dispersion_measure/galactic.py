import os
import numpy
import healpy
import warnings
from astropy import units
from pygedm import generate_healpix_dm_map

_ROOT = os.path.abspath(os.path.dirname(__file__))
_DATA = _ROOT.replace('dispersion_measure', 'data')

unit = units.pc / units.cm**3
galactic_edge = 30 * units.kpc


class Galactic():
    """ """

    def __init__(self, nside=128, method='yt2020_analytic'):

        self.nside = nside
        self.method = method

        if method == 'yt2020':

            warning_message = ''.join([
                'YT2020 method is even slower,',
                ' consider using yt2020_analytic'
            ])

            warnings.warn(warning_message)

    def __call__(self, gl, gb):

        if not hasattr(self, 'dm_map'):
            path = '{}/{}_{}.npy'.format(_DATA, self.method, self.nside)
            if os.path.exists(path):
                self.dm_map = numpy.load(path)
            else:
                self.dm_map = generate_healpix_dm_map(galactic_edge,
                                                      self.nside,
                                                      self.method)
                numpy.save(path, self.dm_map, allow_pickle=True)

        dm_gal = healpy.get_interp_val(self.dm_map, gl.value,
                                       gb.value, lonlat=True)
        return unit * dm_gal
