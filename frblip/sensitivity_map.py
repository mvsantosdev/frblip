import types

import numpy
import xarray
from sparse import COO
from astropy_healpix import HEALPix

from astropy.time import Time

from functools import cached_property

from .random import Redshift, Schechter
from .cosmology import Cosmology

from astropy import units, coordinates

from .basic_sampler import BasicSampler
from .decorators import xarrayfy, default_units
from .decorators import observation_method, todense_option

from .observation import Observation


class SensitivityMap(BasicSampler, HEALPix):

    KIND = 'PIXEL'

    def __init__(
        self,
        nside: int = 128,
        order: str = 'ring',
        phistar: float | units.Quantity = 339,
        gamma: float = -1.79,
        log_Lstar: float | units.Quantity = 44.46,
        log_L0: float | units.Quantity = 41.96,
        low_frequency: float | units.Quantity = 10,
        high_frequency: float | units.Quantity = 10000,
        low_frequency_cal: float | units.Quantity = 400,
        high_frequency_cal: float | units.Quantity = 1400,
        emission_frame: bool = False,
        cosmology: str = 'Planck_18',
        zmin: float = 0.0,
        zmax: float = 30.0,
    ):

        HEALPix.__init__(self, nside, order)

        self._load_params(
            phistar,
            gamma,
            log_Lstar,
            log_L0,
            low_frequency,
            high_frequency,
            low_frequency_cal,
            high_frequency_cal,
            emission_frame,
            cosmology,
            zmin,
            zmax,
        )

    @default_units(
        log_L0='dex(erg / s)',
        log_Lstar='dex(erg / s)',
        phistar='1 / (Gpc^3 yr)',
        low_frequency='MHz',
        high_frequency='MHz',
        low_frequency_cal='MHz',
        high_frequency_cal='MHz',
    )
    def _load_params(
        self,
        phistar: float | units.Quantity,
        gamma: float,
        log_Lstar: float | units.Quantity,
        log_L0: float | units.Quantity,
        low_frequency: float | units.Quantity,
        high_frequency: float | units.Quantity,
        low_frequency_cal: float | units.Quantity,
        high_frequency_cal: float | units.Quantity,
        emission_frame: bool,
        cosmology: str,
        zmin: float,
        zmax: float,
    ):

        self.log_L0 = log_L0
        self.log_Lstar = log_Lstar
        self.phistar = phistar
        self.gamma = gamma
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.low_frequency_cal = low_frequency_cal
        self.high_frequency_cal = high_frequency_cal
        self.emission_frame = emission_frame
        self.cosmology = cosmology
        self.zmin = zmin
        self.zmax = zmax

    @cached_property
    def _cosmology(self) -> Cosmology:
        return Cosmology(self.cosmology)

    @cached_property
    def _xmin(self) -> units.Quantity:
        dlogL = self.log_L0 - self.log_Lstar
        return dlogL.to(1).value

    @cached_property
    def _zdist(self) -> Redshift:
        return Redshift(
            zmin=self.zmin, zmax=self.zmax, cosmology=self._cosmology
        )

    @cached_property
    def _lumdist(self) -> Schechter:
        return Schechter(self._xmin, self.gamma)

    @property
    def size(self) -> int:
        return self.npix

    @cached_property
    def pixels(self) -> numpy.ndarray:
        return numpy.arange(self.npix)

    @cached_property
    def gcrs(self) -> coordinates.GCRS:

        pixels = numpy.arange(self.npix)
        ra, dec = self.healpix_to_lonlat(pixels)
        return coordinates.GCRS(ra=ra, dec=dec)

    @cached_property
    def itrs(self) -> coordinates.ITRS:

        frame = coordinates.ITRS(obstime=self.time)
        return self.gcrs.transform_to(frame)

    @cached_property
    def icrs(self) -> coordinates.ICRS:

        frame = coordinates.ICRS()
        return self.gcrs.transform_to(frame)

    @cached_property
    def xyz(self) -> units.Quantity:

        return self.itrs.cartesian.xyz

    @cached_property
    def time(self) -> Time:

        j2000 = Time('J2000').to_datetime()
        return Time(j2000)

    def luminosity_threshold(
        self,
        redshift: float,
        density_flux: units.Quantity,
        spectral_index: float,
    ) -> units.Quantity:

        sip = 1 + spectral_index
        sign = numpy.sign(sip)
        nuhp = (self.high_frequency_cal / units.MHz) ** sip
        nulp = (self.low_frequency_cal / units.MHz) ** sip
        dnu = sign * (nuhp - nulp)

        lum_dist = self._cosmology.luminosity_distance(redshift.ravel())
        lum_dist = lum_dist.reshape(*redshift.shape)

        Lthre = 4 * numpy.pi * dnu * lum_dist**2 * density_flux
        if self.emission_frame:
            return Lthre / (1 + redshift) ** sip
        return Lthre

    @observation_method
    def altaz(self, observation: Observation) -> coordinates.SkyCoord:

        return getattr(observation, 'altaz', None)

    def get_redshift_rate(
        self,
        sensitivity: xarray.DataArray,
        zmin: float = 0,
        zmax: float = 30,
        spectral_index: float = 0.0,
        total: str | bool = False,
        channels: int = 1,
        unit: units.Unit | str = '1/year',
        eps: float = 1e-4,
    ) -> units.Quantity:

        zg, rates = self.get_redshift_table(
            sensitivity, zmin, zmax, spectral_index, total, channels, unit, eps
        )

        def redshift_rate(z):

            y = numpy.interp(x=z, xp=zg, fp=rates.value)
            return y * rates.unit

        return redshift_rate

    @numpy.errstate(divide='ignore', over='ignore')
    @default_units(time='year')
    def get_maximum_redshift(
        self,
        sensitivity: xarray.DataArray,
        zmin: float = 0,
        zmax: float = 30,
        spectral_index: float = 0.0,
        total: str | bool = False,
        channels: int = 1,
        time: units.Quantity | float | str = '1 year',
        tolerance: float = 1,
        eps: float = 1e-4,
    ) -> tuple[float, float, float]:

        redshift, rates = self.get_redshift_table(
            sensitivity, zmin, zmax, spectral_index, total, channels, eps=eps
        )

        idx = rates.argmax()
        x = redshift[idx:][::-1]
        y = (rates[idx:][::-1] * time).to(1).clip(0)

        log_y = numpy.log(y / tolerance)
        idx = numpy.isfinite(log_y)
        center = numpy.interp(x=0, xp=log_y[idx], fp=x[idx])

        ly = y - 1.96 * numpy.sqrt(y)
        log_ly = numpy.log(ly / tolerance)
        idx = numpy.isfinite(log_ly)
        lower = numpy.interp(x=0, xp=log_ly[idx], fp=x[idx])

        uy = y + 1.96 * numpy.sqrt(y)
        log_uy = numpy.log(uy / tolerance)
        idx = numpy.isfinite(log_uy)
        upper = numpy.interp(x=0, xp=log_uy[idx], fp=x[idx])

        return center, lower, upper

    @observation_method
    @xarrayfy(snr='SNR')
    def maximum_redshift(
        self,
        observation: Observation,
        zmin: float = 0,
        zmax: float = 30,
        spectral_index: float = 0.0,
        snr: xarray.DataArray | numpy.ndarray | range = range(1, 11),
        total: str | bool = False,
        channels: int = 1,
        time: units.Quantity | float | str = '1 year',
        tolerance: float = 1,
        eps: float = 1e-4,
    ) -> tuple[float, float, float]:

        sensitivity = self._sensitivity(
            observation, spectral_index, total, channels
        )
        sens = snr * sensitivity
        sens.attrs['unit'] = sensitivity.unit

        zmin, zmax = self._redshift_range(observation)

        return self.get_maximum_redshift(
            sens,
            zmin,
            zmax,
            spectral_index,
            total,
            channels,
            time,
            tolerance,
            eps,
        )

    def get_redshift_table(
        self,
        sensitivity: xarray.DataArray,
        zmin: float = 0,
        zmax: float = 30,
        spectral_index: float = 0.0,
        total: str | bool = False,
        channels: int = 1,
        unit: units.Unit | str = '1/year',
        eps: float = 1e-4,
    ) -> tuple[numpy.ndarray, units.Quantity]:

        data = sensitivity.data
        if not isinstance(data, COO):
            data = COO(data, fill_value=numpy.inf)
        sflux = data.data * sensitivity.unit

        smin = sflux.min()
        smax = sflux.max()

        zg, sg, table = self.get_rate_table(
            smin, smax, zmin, zmax, spectral_index, eps
        )

        table = (table * self.pixel_area).to(unit)

        rates = numpy.apply_along_axis(
            lambda x: numpy.interp(x=sflux, xp=sg, fp=x).sum(),
            axis=1,
            arr=table,
        )

        return zg.ravel(), rates

    @observation_method
    @xarrayfy(snr='SNR')
    def redshift_table(
        self,
        observation: Observation,
        spectral_index: float = 0.0,
        snr: xarray.DataArray | numpy.ndarray | range = range(1, 11),
        total: str | bool = False,
        channels: int = 1,
        unit: units.Unit | str = '1/year',
        eps: float = 1e-4,
    ) -> tuple[numpy.ndarray, units.Quantity]:

        sensitivity = self._sensitivity(
            observation, spectral_index, total, channels
        )
        sens = snr * sensitivity
        sens.attrs['unit'] = sensitivity.unit

        zmin, zmax = self._redshift_range(observation)

        return self.get_redshift_table(
            sens, zmin, zmax, spectral_index, total, channels, unit, eps
        )

    @numpy.errstate(divide='ignore', over='ignore')
    def get_rate_table(
        self,
        smin: units.QuantityInfo,
        smax: units.QuantityInfo,
        zmin: float,
        zmax: float,
        spectral_index: float = 0.0,
        eps: float = 1e-4,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:

        log_min = units.LogQuantity(smin)
        log_max = units.LogQuantity(smax)

        ngrid = int(1 / eps)
        sgrid = numpy.logspace(log_min, log_max, ngrid)
        zgrid = numpy.linspace(zmin, zmax, ngrid)

        dz = self._zdist.angular_density(zgrid)

        zgrid = zgrid.reshape(-1, 1)
        dz = dz.reshape(-1, 1)

        Lthre = self.luminosity_threshold(zgrid, sgrid, spectral_index)
        xthre = Lthre.to(self.log_Lstar.unit) - self.log_Lstar
        xmin, xmax = self._lumdist.xmin, self._lumdist.xmax
        xthre = xthre.to(1).value.clip(xmin, xmax)

        norm = self.phistar / self._lumdist.pdf_norm
        lum_integral = norm * self._lumdist.sf(xthre)

        pdf = lum_integral * dz

        return zgrid, sgrid, pdf

    def specific_rate(
        self,
        smin: units.QuantityInfo,
        smax: units.QuantityInfo,
        zmin: float,
        zmax: float,
        unit: units.Unit | str,
        spectral_index: float,
        eps: float = 1e-4,
    ) -> types.FunctionType:

        zg, sg, pdf = self.get_rate_table(
            smin, smax, zmin, zmax, spectral_index, eps
        )
        spdf = numpy.trapz(x=zg, y=pdf, axis=0)
        spdf = spdf * self.pixel_area

        xp, fp = sg, spdf.to(unit)

        def specific_rate(x):

            y = numpy.interp(x=x, xp=xp, fp=fp.value)
            return y * fp.unit

        return specific_rate

    def _redshift_range(
        self,
        observation: Observation,
        channels: int = 1,
    ) -> tuple[float, float]:

        return observation.redshift_range(
            self.low_frequency, self.high_frequency
        )

    @observation_method
    def redshift_range(
        self,
        observation: Observation,
        channels: int = 1,
    ) -> tuple[float, float]:

        return self._redshift_range(observation, channels)

    def _noise(
        self,
        observation: Observation,
        total: str | bool = False,
        channels: int = 1,
    ) -> xarray.DataArray:

        return observation.get_noise(total, channels)

    @observation_method
    @todense_option
    def noise(
        self,
        observation: Observation,
        total: str | bool = False,
        channels: int = 1,
        todense: bool = False,
    ) -> xarray.DataArray:

        return self._noise(observation, total, channels)

    def _sensitivity(
        self,
        observation: Observation,
        spectral_index: float = 0.0,
        total: str | bool = False,
        channels: int = 1,
    ) -> xarray.DataArray:

        sign = numpy.sign(spectral_index + 1)
        spec_idx = numpy.full(self.size, spectral_index)
        freq_resp = observation.get_frequency_response(spec_idx, channels)
        noise = self._noise(observation, total, channels)

        sensitivity = sign * noise / freq_resp
        sensitivity.attrs['unit'] = noise.unit / freq_resp.unit
        return sensitivity

    @observation_method
    @todense_option
    def sensitivity(
        self,
        observation: Observation,
        spectral_index: float = 0.0,
        total: str | bool = False,
        channels: int = 1,
        todense: bool = False,
    ) -> xarray.DataArray:

        return self._sensitivity(observation, spectral_index, total, channels)

    @numpy.errstate(over='ignore')
    def get_rate_map(
        self,
        sensitivity: xarray.DataArray,
        zmin: float = 0,
        zmax: float = 30,
        spectral_index: float = 0.0,
        unit: units.Unit | str = 'year',
        eps: float = 1e-4,
    ) -> xarray.DataArray:

        if isinstance(unit, str):
            unit = units.Unit(unit)
        elif isinstance(unit, units.Quantity):
            unit = unit.unit

        assert unit.physical_type == 'time', f'{unit} is not a time unit'

        rate_unit = 1 / unit

        unit = sensitivity.unit
        data = sensitivity.data
        if not isinstance(data, COO):
            data = COO(data, fill_value=numpy.inf)
        sflux = data.data * unit

        smin = sflux.min()
        smax = sflux.max()

        spec_rate = self.specific_rate(
            smin, smax, zmin, zmax, rate_unit, spectral_index, eps
        )
        rates = spec_rate(sflux)

        rate_map = COO(data.coords, rates, data.shape)
        rate_map = xarray.DataArray(rate_map, dims=sensitivity.dims)
        rate_map.attrs['unit'] = rates.unit

        return rate_map

    def get_rate(
        self,
        sensitivity: xarray.DataArray,
        zmin: float = 0,
        zmax: float = 30,
        spectral_index: float = 0.0,
        unit: units.Unit | str = 'year',
        eps: float = 1e-4,
    ) -> xarray.DataArray:

        rate_map = self.get_rate_map(
            sensitivity, zmin, zmax, spectral_index, unit, eps
        )
        return rate_map.sum('PIXEL', keep_attrs=True)

    @xarrayfy(snr=('SNR',))
    def _si_rate_map(
        self,
        observation: Observation,
        spectral_index: float = 0.0,
        snr: xarray.DataArray | numpy.ndarray | range = range(1, 11),
        total: str | bool = False,
        channels: int = 1,
        unit: units.Unit | str = 'year',
        eps: float = 1e-4,
    ) -> xarray.DataArray:

        sensitivity = self._sensitivity(
            observation, spectral_index, total, channels
        )

        sens = sensitivity * snr
        sens.attrs = sensitivity.attrs

        zmin, zmax = self._redshift_range(observation)

        return self.get_rate_map(sens, zmin, zmax, spectral_index, unit, eps)

    def _rate_map(
        self,
        observation: Observation,
        spectral_index: float = 0.0,
        snr: xarray.DataArray | numpy.ndarray | range = range(1, 11),
        total: str | bool = False,
        channels: int = 1,
        unit: units.Unit | str = 'year',
        eps: float = 1e-4,
    ) -> xarray.DataArray:

        spec_idxs = numpy.atleast_1d(spectral_index)

        rates = xarray.concat(
            [
                self._si_rate_map(
                    observation,
                    spectral_index,
                    snr,
                    total,
                    channels,
                    unit,
                    eps,
                )
                for spec_idx in spec_idxs
            ],
            dim='Spectral Index',
        )

        rates = rates.assign_coords({'SNR': snr})

        return rates.squeeze()

    @observation_method
    @todense_option
    def rate_map(
        self,
        observation: Observation,
        spectral_index: Observation = 0.0,
        snr: xarray.DataArray | numpy.ndarray | range = range(1, 11),
        total: str | bool = False,
        channels: int = 1,
        unit: units.Unit | str = 'year',
        eps: float = 1e-4,
        todense: bool = False,
    ) -> xarray.DataArray:

        return self._rate_map(
            observation, spectral_index, snr, total, channels, unit, eps
        )

    def _rate(
        self,
        observation: Observation,
        spectral_index: float = 0.0,
        snr: xarray.DataArray | numpy.ndarray | range = range(1, 11),
        total: str | bool = False,
        channels: int = 1,
        unit: units.Unit | str = 'year',
        eps: float = 1e-4,
    ) -> xarray.DataArray:

        rate_map = self._rate_map(
            observation, spectral_index, snr, total, channels, unit, eps
        )
        return rate_map.sum('PIXEL', keep_attrs=True)

    @observation_method
    @todense_option
    def rate(
        self,
        observation: Observation,
        spectral_index: float = 0.0,
        snr: xarray.DataArray | numpy.ndarray | range = range(1, 11),
        total: str | bool = False,
        channels: int = 1,
        unit: units.Unit | str = 'year',
        eps: float = 1e-4,
        todense: bool = True,
    ) -> xarray.DataArray:

        return self._rate(
            observation, spectral_index, snr, total, channels, unit, eps
        )
