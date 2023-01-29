import numpy

from astropy import units, coordinates

from .basic_sampler import BasicSampler


class UVGrid(BasicSampler):

    def __init__(self,
                 urange: tuple[float, float] = (-1, 1),
                 vrange: tuple[float, float] = (-1, 1),
                 nu=201, nv=201):

        self.urange = urange
        self.vrange = vrange

        self.nu = nu
        self.nv = nv

        u = numpy.linspace(*urange, nu)
        v = numpy.linspace(*vrange, nv)

        self.u, self.v = numpy.meshgrid(u, v)

        self.kind = 'POINT'

    @property
    def size(self):

        return self.nu * self.nv

    def polar(self):

        q = numpy.arctan2(self.v, self.u)
        r = numpy.sqrt(self.v**2 + self.u**2)

        return q, r

    @numpy.errstate(invalid='ignore')
    def altaz_from_location(self, location=None, interp=300):

        u = self.u.ravel()
        v = self.v.ravel()
        w = numpy.sqrt(1 - u**2 - v**2)

        _, alt, az = coordinates.cartesian_to_spherical(u, v, w)

        az = az.to(units.deg)
        alt = numpy.nan_to_num(alt, nan=-90*units.deg).to(units.deg)

        return coordinates.AltAz(alt=alt, az=az)

    def _sensitivity(self, name, spectral_index=0.0, total=False, channels=1):

        observation = self[name]
        spec_idx = numpy.full(self.size, spectral_index)
        freq_resp = observation.get_frequency_response(spec_idx, channels)
        noise = self._noise(name, total, channels)

        sensitivity = noise / freq_resp
        sensitivity.attrs['unit'] = noise.unit / freq_resp.unit
        return sensitivity

    def _noise(self, name, total=False, channels=1):

        observation = self[name]
        return observation.get_noise(total, channels)
