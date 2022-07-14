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

from .basic_sampler import BasicSampler


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
        return itemgetter(*keys)(self.observations)

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

    def _response(self, name, channels=1, spectral_index=0.0):

        observation = self[name]
        si = numpy.full(self.npix, spectral_index)
        peak_density_flux = observation.get_frequency_response(si, channels)
        return observation.response * peak_density_flux

    def _noise(self, name, channels=1):

        observation = self[name]
        return observation.get_noise(channels)

    @numpy.errstate(divide='ignore', over='ignore')
    def _sensitivity(self, name, channels=1, spectral_index=0.0,
                     total=False, level=None):

        noise = self._noise(name, channels)
        response = self._response(name, channels, spectral_index)
        sensitivity = (1 / response) * noise

        if total:
            lvl = [
                dim for dim in sensitivity.dims
                if dim not in ('PIXEL', 'CHANNEL')
            ]
            sensitivity = sensitivity.min(lvl)
        if level is not None:
            sensitivity = sensitivity.min(level)
        sensitivity = sensitivity.squeeze()
        sensitivity.attrs = noise.attrs
        return sensitivity

    @numpy.errstate(divide='ignore', over='ignore')
    def __specific_rate(self, smin, smax, zmin, zmax, frequency_range,
                        unit, spectral_index, eps=1e-4):

        log_min = units.LogQuantity(smin)
        log_max = units.LogQuantity(smax)

        ngrid = int(1 / eps)
        sgrid = numpy.logspace(log_min, log_max, ngrid)
        zgrid = numpy.linspace(zmin, zmax, ngrid)

        lum_dist = self.__cosmology.luminosity_distance(zgrid)
        dz = self.__zdist.angular_density(zgrid)

        dz = dz[:, numpy.newaxis]
        zgrid = zgrid[:, numpy.newaxis]
        lum_dist = lum_dist[:, numpy.newaxis]

        _sip = 1 + spectral_index
        nuhp = (self.high_frequency_cal / units.MHz)**_sip
        nulp = (self.low_frequency_cal / units.MHz)**_sip
        dnup = (frequency_range / units.MHz)**_sip
        width = frequency_range.diff()
        nu_factor = width * (nuhp - nulp) / dnup.diff(axis=-1)

        Lthre = 4 * numpy.pi * nu_factor * lum_dist**2 * sgrid
        if self.emission_frame:
            Lthre = Lthre / (1 + zgrid)**_sip

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

    @numpy.errstate(over='ignore')
    def _si_rate_map(self, name=None, channels=1, sensitivity=None,
                     snr=None, unit='year', spectral_index=0.0,
                     total=False, eps=1e-4):

        if isinstance(unit, str):
            unit = units.Unit(unit)
        elif isinstance(unit, units.Quantity):
            unit = unit.unit

        assert unit.physical_type == 'time', \
               '{} is not a time unit'.format(unit.to_string())

        rate_unit = 1 / unit

        s = numpy.arange(1, 11) if snr is None else snr
        s = xarray.DataArray(numpy.atleast_1d(s), dims='SNR')

        if not isinstance(sensitivity, numpy.ndarray):
            sensitivity = self._sensitivity(name, channels,
                                            spectral_index,
                                            total)

        unit = sensitivity.attrs['unit']
        sensitivity = sensitivity * s

        data = sensitivity.data
        sflux = data.data * unit

        smin = sflux.min()
        smax = sflux.max()

        frequency_range = self[name].frequency_range
        zmax = self.high_frequency / frequency_range[0] - 1
        zmin = self.low_frequency / frequency_range[-1] - 1
        zmin = zmin.clip(0)

        specific_rate = self.__specific_rate(smin, smax, zmin, zmax,
                                             frequency_range, rate_unit,
                                             spectral_index, eps)

        rates = specific_rate(sflux)

        rate_map = COO(data.coords, rates, data.shape)
        rate_map = xarray.DataArray(rate_map, dims=sensitivity.dims)
        rate_map.attrs['unit'] = rates.unit

        return rate_map

    def _rate_map(self, name=None, channels=1, sensitivity=None, snr=None,
                  spectral_index=0.0, total=False, unit='year', eps=1e-4):

        sis = numpy.atleast_1d(spectral_index)

        rates = xarray.concat([
            self._si_rate_map(name, channels=channels, snr=snr,
                              sensitivity=sensitivity, total=total,
                              spectral_index=si, unit=unit, eps=eps)
            for si in sis
        ], dim='Spectral Index')

        return rates.squeeze()

    def _rate(self, name=None, channels=1, sensitivity=None, snr=None,
              spectral_index=0.0, total=False, unit='year', eps=1e-4):

        return self._rate_map(name, channels=channels, snr=snr,
                              sensitivity=sensitivity, total=total,
                              spectral_index=spectral_index, unit=unit,
                              eps=eps).sum('PIXEL', keep_attrs=True)
