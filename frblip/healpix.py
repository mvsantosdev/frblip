import numpy
import xarray
from sparse import COO
from astropy_healpix import HEALPix

from astropy.time import Time

from operator import itemgetter
from functools import cached_property

from .random import Redshift, Schechter
from .cosmology import Cosmology, builtin

from astropy import units, coordinates

from .basic_sampler import getufunc, BasicSampler


class HealPixMap(BasicSampler, HEALPix):

    def __init__(self, nside=128, order='ring', phistar=339,
                 alpha=-1.79, log_Lstar=44.46, log_L0=41.96,
                 low_frequency=10, high_frequency=10000,
                 low_frequency_cal=400, high_frequency_cal=1400,
                 emission_frame=False, cosmology='Planck_18',
                 zmin=0.0, zmax=30.0):

        HEALPix.__init__(self, nside, order)

        self.__load_params(phistar, alpha, log_Lstar, log_L0,
                           low_frequency, high_frequency,
                           low_frequency_cal, high_frequency_cal,
                           emission_frame, cosmology, zmin, zmax)

        self.kind = 'PIXEL'

    def __load_params(self, phistar, alpha, log_Lstar, log_L0,
                      low_frequency, high_frequency,
                      low_frequency_cal, high_frequency_cal,
                      emission_frame, cosmology, zmin, zmax):

        self.log_L0 = log_L0 * units.LogUnit(units.erg / units.s)
        self.log_Lstar = log_Lstar * units.LogUnit(units.erg / units.s)
        self.phistar = phistar / (units.Gpc**3 * units.year)
        self.alpha = alpha
        self.low_frequency = low_frequency * units.MHz
        self.high_frequency = high_frequency * units.MHz
        self.low_frequency_cal = low_frequency_cal * units.MHz
        self.high_frequency_cal = high_frequency_cal * units.MHz
        self.emission_frame = emission_frame
        self.cosmology = cosmology
        self.zmin = zmin
        self.zmax = zmax

    def __getitem__(self, keys):

        if isinstance(keys, str):
            return self.observations[keys]
        keys = numpy.array(keys)
        if numpy.issubdtype(keys.dtype, numpy.str_):
            return itemgetter(*keys)(self.observations)
        return None

    @cached_property
    def __cosmology(self):
        params = builtin[self.cosmology]
        return Cosmology(**params)

    @cached_property
    def __xmin(self):
        dlogL = self.log_L0 - self.log_Lstar
        return dlogL.to(1).value

    @cached_property
    def __zdist(self):
        return Redshift(zmin=self.zmin, zmax=self.zmax,
                        cosmology=self.__cosmology)

    @cached_property
    def __lumdist(self):
        return Schechter(self.__xmin, self.alpha)

    @property
    def size(self):
        return self.npix

    @cached_property
    def pixels(self):
        return numpy.arange(self.npix)

    @cached_property
    def gcrs(self):
        pixels = numpy.arange(self.npix)
        ra, dec = self.healpix_to_lonlat(pixels)
        return coordinates.GCRS(ra=ra, dec=dec)

    @cached_property
    def itrs(self):
        frame = coordinates.ITRS(obstime=self.itrs_time)
        return self.gcrs.transform_to(frame)

    @cached_property
    def icrs(self):
        frame = coordinates.ICRS()
        return self.gcrs.transform_to(frame)

    @cached_property
    def xyz(self):
        return self.itrs.cartesian.xyz

    @cached_property
    def itrs_time(self):
        j2000 = Time('J2000').to_datetime()
        return Time(j2000)

    def luminosity_threshold(self, redshift, density_flux, spectral_index):

        sip = 1 + spectral_index
        nuhp = (self.high_frequency_cal / units.MHz)**sip
        nulp = (self.low_frequency_cal / units.MHz)**sip

        lum_dist = self.__cosmology.luminosity_distance(redshift.ravel())
        lum_dist = lum_dist.reshape(*redshift.shape)
        Lthre = 4 * numpy.pi * (nuhp - nulp) * lum_dist**2 * density_flux
        if self.emission_frame:
            return Lthre / (1 + redshift)**sip
        return Lthre

    @numpy.errstate(divide='ignore', over='ignore')
    def specific_rate(self, smin, smax, zmin, zmax, unit,
                      spectral_index, eps=1e-4):

        log_min = units.LogQuantity(smin)
        log_max = units.LogQuantity(smax)

        ngrid = int(1 / eps)
        sgrid = numpy.logspace(log_min, log_max, ngrid)
        zgrid = numpy.linspace(zmin, zmax, ngrid)

        dz = self.__zdist.angular_density(zgrid)

        zgrid = zgrid.reshape(-1, 1)
        dz = dz.reshape(-1, 1)

        Lthre = self.luminosity_threshold(zgrid, sgrid, spectral_index)
        xthre = Lthre.to(self.log_Lstar.unit) - self.log_Lstar
        xmin, xmax = self.__lumdist.xmin, self.__lumdist.xmax
        xthre = xthre.to(1).value.clip(xmin, xmax)

        norm = self.phistar / self.__lumdist.pdf_norm
        lum_integral = norm * self.__lumdist.sf(xthre)

        integrand = lum_integral * dz
        zintegral = numpy.trapz(x=zgrid, y=integrand, axis=0)
        zintegral = zintegral * self.pixel_area

        xp, fp = sgrid, zintegral.to(unit)

        def specific_rate(x):
            y = numpy.interp(x=x, xp=xp, fp=fp.value)
            return y * fp.unit

        return specific_rate

    def _response(self, name, channels=1, spectral_index=0.0):

        observation = self[name]
        spec_idx = numpy.full(self.size, spectral_index)
        freq_resp = observation.get_frequency_response(spec_idx, channels)
        response = observation.response * freq_resp
        response.attrs = freq_resp.attrs
        return response

    def _noise(self, name, channels=1):

        observation = self[name]
        return observation.get_noise(channels)

    def _response_to_noise(self, name, channels=1, spectral_index=0.0,
                           total=False, method='max', **kwargs):

        func = getufunc(method, **kwargs)
        response = self._response(name, channels, spectral_index)
        noise = self._noise(name, channels)
        resp_to_noise = response / noise

        if isinstance(total, str) and (total in resp_to_noise.dims):
            resp_to_noise = resp_to_noise.reduce(func, dim=total, **kwargs)
        if isinstance(total, list):
            levels = [*filter(lambda x: x in response.dims, total)]
            resp_to_noise = resp_to_noise.reduce(func, dim=levels, **kwargs)
        elif total is True:
            levels = [*filter(lambda x: x not in (self.kind, 'CHANNEL'),
                              resp_to_noise.dims)]
            resp_to_noise = resp_to_noise.reduce(func, dim=levels, **kwargs)

        resp_to_noise.attrs['unit'] = response.unit / noise.unit
        return resp_to_noise

    def _sensitivity(self, name, channels=1, spectral_index=0.0,
                     total=False, method='max', **kwargs):

        resp_to_noise = self._response_to_noise(name, channels, spectral_index,
                                                total, method, **kwargs)
        sensitivity = 1 / resp_to_noise
        sensitivity.attrs['unit'] = 1 / resp_to_noise.unit
        sensitivity.attrs['unit'] = sensitivity.unit.to(units.MHz * units.Jy)
        return sensitivity

    def _redshift_range(self, name, channels=1):

        frequency_range = self[name].frequency_range
        zmax = self.high_frequency / frequency_range[0] - 1
        zmin = self.low_frequency / frequency_range[-1] - 1
        zmin = zmin.clip(0)

        return zmin.value, zmax.value

    @numpy.errstate(over='ignore')
    def get_rate_map(self, sensitivity, unit='year', zmin=0, zmax=30,
                     spectral_index=0.0, eps=1e-4):

        if isinstance(unit, str):
            unit = units.Unit(unit)
        elif isinstance(unit, units.Quantity):
            unit = unit.unit

        assert unit.physical_type == 'time', \
               '{} is not a time unit'.format(unit.to_string())

        rate_unit = 1 / unit

        unit = sensitivity.unit
        data = sensitivity.data
        if not isinstance(data.data, COO):
            data = COO(data, fill_value=numpy.inf)
        sflux = data.data * unit

        smin = sflux.min()
        smax = sflux.max()

        spec_rate = self.specific_rate(smin, smax, zmin, zmax, rate_unit,
                                       spectral_index, eps)
        rates = spec_rate(sflux)

        rate_map = COO(data.coords, rates, data.shape)
        rate_map = xarray.DataArray(rate_map, dims=sensitivity.dims)
        rate_map.attrs['unit'] = rates.unit

        return rate_map

    def get_rate(self, sensitivity, unit='year', zmin=0, zmax=30,
                 spectral_index=0.0, eps=1e-4):

        rate_map = self.get_rate_map(sensitivity, unit, zmin, zmax,
                                     spectral_index, eps)
        return rate_map.sum('PIXEL')

    def _si_rate_map(self, name=None, channels=1, sensitivity=None,
                     snr=None, unit='year', spectral_index=0.0,
                     eps=1e-4, total=False, method='max', **kwargs):

        s = numpy.arange(1, 11) if snr is None else snr
        s = xarray.DataArray(numpy.atleast_1d(s), dims='SNR')

        sensitivity = self._sensitivity(name, channels, spectral_index,
                                        total, method, **kwargs)

        sens = sensitivity * s
        sens.attrs = sensitivity.attrs

        zmin, zmax = self._redshift_range(name)

        return self.get_rate_map(sens, unit='year', zmin=zmin, zmax=zmax,
                                 spectral_index=0.0, eps=1e-4)

    def _rate_map(self, name=None, channels=1, sensitivity=None, snr=None,
                  unit='year', spectral_index=0.0, eps=1e-4, total=False,
                  method='max', **kwargs):

        spec_idxs = numpy.atleast_1d(spectral_index)

        rates = xarray.concat([
            self._si_rate_map(name, channels, sensitivity, snr, unit,
                              spec_idx, eps, total, method, **kwargs)
            for spec_idx in spec_idxs
        ], dim='Spectral Index')

        return rates.squeeze()

    def _rate(self, name=None, channels=1, sensitivity=None, snr=None,
              unit='year', spectral_index=0.0, eps=1e-4, total=False,
              method='max', **kwargs):

        rate_map = self._rate_map(name, channels, sensitivity, snr, unit,
                                  spectral_index, eps, total, method, **kwargs)
        return rate_map.sum('PIXEL', keep_attrs=True)
