from __future__ import annotations

import bz2
import os
import sys
import types
import warnings
from functools import cached_property
from operator import itemgetter

import dill
import numpy
import sparse
import xarray
from astropy import constants, coordinates, units
from astropy.time import Time
from numpy import random
from scipy._lib._util import check_random_state
from scipy.special import erf
from toolz.dicttoolz import keyfilter, merge, valmap

from .basic_sampler import BasicSampler
from .cosmology import Cosmology
from .decorators import (
    default_units,
    observation_method,
    todense_option,
    xarrayfy,
)
from .observation import Observation
from .random import Redshift, Schechter, SpectralIndex
from .random.dispersion_measure import (
    GalacticDM,
    HostGalaxyDM,
    InterGalacticDM,
)


class FastRadioBursts(BasicSampler):

    KIND = 'FRB'

    def __init__(
        self,
        size: int | None = None,
        duration: units.Quantity | float | str = 1,
        log_Lstar: units.Quantity | float | str = 44.46,
        log_L0: units.Quantity | float | str = 41.96,
        phistar: units.Quantity | float | str = 339,
        gamma: float = -1.79,
        pulse_width: tuple[float, float] = (-6.917, 0.824),
        zmin: float = 0.0,
        zmax: float = 30.0,
        ra_range: units.Quantity | tuple[float, float] = (0, 24),
        dec_range: units.Quantity | tuple[float, float] = (-90, 90),
        start: Time | None = None,
        low_frequency: units.Quantity | float | str = 10.0,
        high_frequency: units.Quantity | float | str = 10000.0,
        low_frequency_cal: units.Quantity | float | str = 400.0,
        high_frequency_cal: units.Quantity | float | str = 1400.0,
        emission_frame: bool = False,
        spectral_index: str = 'CHIME2021',
        gal_method: str = 'yt2020_analytic',
        gal_nside: int = 128,
        host_dist: str = 'lognormal',
        host_source: str = 'luo18',
        host_model: tuple[str, str] = ('ALG', 'YMW16'),
        cosmology: str = 'Planck_18',
        igm_model: str = 'Takahashi2021',
        free_electron_bias: str = 'Takahashi2021',
        random_state: int | numpy.random.RandomState | None = None,
        verbose: bool = True,
    ):

        old_target = sys.stdout
        sys.stdout = old_target if verbose else open(os.devnull, 'w')

        self._load_params(
            duration,
            log_Lstar,
            log_L0,
            phistar,
            gamma,
            pulse_width,
            zmin,
            zmax,
            ra_range,
            dec_range,
            start,
            low_frequency,
            high_frequency,
            low_frequency_cal,
            high_frequency_cal,
            emission_frame,
            spectral_index,
            gal_method,
            gal_nside,
            host_dist,
            host_source,
            host_model,
            cosmology,
            igm_model,
            free_electron_bias,
        )

        self.random_state = check_random_state(random_state).get_state()
        numpy.random.set_state(self.random_state)

        self._frb_rate(size)

        self.params = [*self.__dict__.keys()]

        self._energy_normalization

        sys.stdout = old_target

    @default_units(
        duration='day',
        log_L0='dex(erg s^-1)',
        log_Lstar='dex(erg s^-1)',
        phistar='Gpc^-3 yr^-1',
        ra_range='hourangle',
        dec_range='deg',
        low_frequency='MHz',
        high_frequency='MHz',
        low_frequency_cal='MHz',
        high_frequency_cal='MHz',
    )
    def _load_params(
        self,
        duration: units.Quantity | float | str,
        log_Lstar: units.Quantity | float | str,
        log_L0: units.Quantity | float | str,
        phistar: units.Quantity | float | str,
        gamma: float,
        pulse_width: tuple[float, float],
        zmin: float,
        zmax: float,
        ra_range: units.Quantity | tuple[float, float],
        dec_range: units.Quantity | tuple[float, float],
        start: Time | None,
        low_frequency: units.Quantity | float | str,
        high_frequency: units.Quantity | float | str,
        low_frequency_cal: units.Quantity | float | str,
        high_frequency_cal: units.Quantity | float | str,
        emission_frame: bool,
        spectral_index: str,
        gal_method: str,
        gal_nside: int,
        host_dist: str,
        host_source: str,
        host_model: tuple[str, str],
        cosmology: str,
        igm_model: str,
        free_electron_bias: str,
    ):

        self.duration = duration
        self.zmin = zmin
        self.zmax = zmax
        self.log_L0 = log_L0
        self.log_Lstar = log_Lstar
        self.phistar = phistar
        self.gamma = gamma
        self.w_mean, self.w_std = pulse_width
        self._spec_idx_dist = SpectralIndex(spectral_index)
        self.ra_range = ra_range
        self.dec_range = dec_range
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.low_frequency_cal = low_frequency_cal
        self.high_frequency_cal = high_frequency_cal
        self.igm_model = igm_model
        self.free_electron_bias = free_electron_bias
        self.cosmology = cosmology
        self.host_dist = host_dist
        self.host_source = host_source
        self.host_model = host_model
        self.emission_frame = emission_frame

        if start is None:
            now = Time.now()
            today = now.iso.split()
            self.start = Time(today[0])
        else:
            self.start = start

        self.gal_nside = gal_nside
        self.gal_method = gal_method

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self,
        index: int | str | slice | numpy.ndarray,
    ) -> FastRadioBursts | None:

        if isinstance(index, str):
            return self.observations[index]
        if isinstance(index, slice):
            return self.select(index, inplace=False)

        idx = numpy.array(index)
        numeric = numpy.issubdtype(idx.dtype, numpy.signedinteger)
        boolean = numpy.issubdtype(idx.dtype, numpy.bool_)
        if numeric or boolean:
            return self.select(idx, inplace=False)
        if numpy.issubdtype(idx.dtype, numpy.str_):
            return itemgetter(*idx)(self.observations)
        return None

    def select(
        self,
        index: int | str | slice | numpy.ndarray,
        inplace: bool = False,
    ) -> FastRadioBursts | None:

        if not inplace:
            mock = self.copy(clear=False)
            mock.select(index, inplace=True)
            return mock

        self.__dict__.update(
            {
                name: attr[index]
                for name, attr in self.__dict__.items()
                if hasattr(attr, 'size') and numpy.size(attr) == self.size
            }
        )

        if hasattr(self, 'observations'):
            self.observations.update(
                {
                    name: observation[index]
                    for name, observation in self.observations.items()
                }
            )

        self.size = self.redshift.size

    def iterfrbs(
        self,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> FastRadioBursts:

        stop = self.size if stop is None else stop
        for i in range(start, stop, step):
            yield self[i]

    def iterchunks(
        self,
        size: int = 1,
        start: int = 0,
        stop: int | None = None,
        retindex: bool = False,
    ) -> FastRadioBursts:

        stop = self.size if stop is None else stop

        if retindex:
            for i in range(start, stop, size):
                j = i + size
                yield i, j, self[i:j]
        else:
            for i in range(start, stop, size):
                j = i + size
                yield self[i:j]

    @cached_property
    @default_units('pc^2 MHz')
    def disperse_constant(self):

        e = constants.e.emu
        c = constants.c
        m_e = constants.m_e

        return e**2 * c / m_e / 2 / numpy.pi

    @cached_property
    def _cosmology(self) -> Cosmology:
        return Cosmology(
            source=self.cosmology, free_electron_bias=self.free_electron_bias
        )

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

    @cached_property
    @default_units('day^-1')
    def sky_rate(self) -> units.Quantity:
        Lum = self.phistar / self._lumdist.pdf_norm
        Vol = 1 / self._zdist.pdf_norm
        return Lum * Vol

    @cached_property
    def redshift(self) -> numpy.ndarray:

        return self._zdist.rvs(size=self.size)

    @cached_property
    @default_units('dex(erg s^-1)')
    def log_luminosity(self) -> units.Quantity:

        loglum = self._lumdist.log_rvs(size=self.size)
        return loglum * units.LogUnit() + self.log_Lstar

    @cached_property
    @default_units('ms')
    def pulse_width(self) -> units.Quantity:

        width = random.lognormal(self.w_mean, self.w_std, size=self.size)
        return width * units.s

    @cached_property
    def emitted_pulse_width(self) -> units.Quantity:

        return self.pulse_width / (1 + self.redshift)

    @cached_property
    def time(self) -> Time:

        dt = random.random(size=self.size) * self.duration
        return self.start + numpy.sort(dt)

    @cached_property
    def spectral_index(self) -> numpy.ndarray:

        return self._spec_idx_dist.rvs(self.size)

    @cached_property
    def icrs(self) -> coordinates.SkyCoord:

        sin = numpy.sin(self.dec_range)
        args = random.uniform(*sin, self.size) * sin.unit
        decs = numpy.arcsin(args).to('deg')
        ras = random.uniform(*self.ra_range.value, self.size)
        ras = ras * self.ra_range.unit
        return coordinates.SkyCoord(ras, decs, frame='icrs')

    @cached_property
    def area(self) -> units.Quantity:

        x = numpy.sin(self.dec_range).diff().item()
        y = self.ra_range.to('rad').diff().item()
        Area = (x * y) * units.rad
        return Area.to('deg^2')

    @cached_property
    @default_units('Mpc')
    def luminosity_distance(self) -> units.Quantity:
        z = self.redshift
        return self._cosmology.luminosity_distance(z)

    @cached_property
    @default_units('erg s^-1')
    def _luminosity(self) -> units.Quantity:
        return self.log_luminosity

    @cached_property
    @default_units('Jy MHz')
    def flux(self) -> units.Quantity:

        surface = 4 * numpy.pi * self.luminosity_distance**2
        return self._luminosity / surface

    @cached_property
    def _energy_normalization(self) -> units.Quantity:
        _sip1 = self.spectral_index + 1
        nu_lp = (self.low_frequency_cal / units.MHz) ** _sip1
        nu_hp = (self.high_frequency_cal / units.MHz) ** _sip1
        sflux = self.flux / (nu_hp - nu_lp)
        if self.emission_frame:
            z_factor = (1 + self.redshift) ** _sip1
            return sflux * z_factor
        return sflux

    @cached_property
    def itrs(self) -> coordinates.SkyCoord:

        itrs_frame = coordinates.ITRS(obstime=self.time)
        return self.icrs.transform_to(itrs_frame)

    @property
    def xyz(self) -> units.Quantity:

        return self.itrs.cartesian.xyz

    @property
    def galactic(self) -> units.Quantity:

        return self.icrs.galactic

    @cached_property
    def _gal_dm(self) -> GalacticDM:
        return GalacticDM(self.gal_nside, self.gal_method)

    @cached_property
    def _igm_dm(self) -> InterGalacticDM:
        return InterGalacticDM(
            free_electron_model=self.igm_model, cosmology=self._cosmology
        )

    @cached_property
    def _host_dm(self) -> HostGalaxyDM:
        return HostGalaxyDM(
            self.host_source, self.host_model, self._cosmology, self.host_dist
        )

    @cached_property
    def galactic_dm(self) -> units.Quantity:

        gl = self.galactic.l
        gb = self.galactic.b
        return self._gal_dm(gl, gb)

    @cached_property
    def igm_dm(self) -> units.Quantity:

        z = self.redshift
        return self._igm_dm(z)

    @cached_property
    def host_dm(self) -> units.Quantity:

        z = self.redshift
        return self._host_dm(z)

    @cached_property
    def extra_galactic_dm(self) -> units.Quantity:

        z = self.redshift
        igm = self.igm_dm
        host = self.host_dm
        return igm + host / (1 + z)

    @cached_property
    def dispersion_measure(self) -> units.Quantity:

        return self.galactic_dm + self.extra_galactic_dm

    def clear_cache(self):

        deriveds = ['luminosity_distance', 'flux', '_energy_normalization']

        for derived in deriveds:
            del self.__dict__[derived]

    def _frb_rate(self, size: int | None = None):

        if isinstance(size, int):

            self.size = size
            self.rate = size / self.duration

        elif size is None:

            dec_diff = numpy.sin(self.dec_range).diff() * units.rad
            ra_diff = self.ra_range.to('rad').diff()
            area = (dec_diff * ra_diff).item()
            self.area = area.to('deg^2')

            sky_fraction = (self.area / units.spat).to(1)

            rate = self.sky_rate * sky_fraction

            rate = numpy.round(rate, 0)

            new_size = (rate * self.duration).to(1).value
            self.size = int(new_size)
            self.rate = rate

        else:

            raise TypeError('size must be an integer or None.')

        self.rate = self.rate.to('1 / day')

    def update(self):

        self.time = self.time + self.duration
        kw = {
            'x': self.itrs.x,
            'y': self.itrs.y,
            'z': self.itrs.z,
            'obstime': self.time,
        }
        self.itrs = coordinates.ITRS(**kw)

        if isinstance(self.altaz, coordinates.SkyCoord):
            kw = {
                'alt': self.altaz.alt,
                'az': self.altaz.az,
                'obstime': self.altaz.obstime + self.duration,
            }
            self.altaz = coordinates.AltAz(**kw)
        elif 'observations' in dir(self):
            for name in self.observations:
                self.observations[name].update(self.duration)

    def shuffle(
        self,
        update: bool = True,
        inplace: bool = True,
        skip: list | tuple = ('icrs', 'itrs', 'time', 'altaz', 'observations'),
        full: bool = True,
    ) -> FastRadioBursts:

        warnings.filterwarnings(
            'ignore', category=numpy.VisibleDeprecationWarning
        )

        copy = self if inplace else self.copy()

        params = (*copy.params, *skip, 'params')
        idx = numpy.arange(copy.size)

        if full:

            copy.__dict__.update(
                {
                    key: value[random.choice(idx, idx.size, False)]
                    for key, value in copy.__dict__.items()
                    if numpy.size(value) == copy.size and key not in skip
                }
            )

        else:

            idx = random.choice(idx, idx.size, False)

            copy.__dict__.update(
                {
                    key: value[idx]
                    for key, value in copy.__dict__.items()
                    if numpy.size(value) == copy.size and key not in params
                }
            )

        if update:
            copy.update()

        return copy

    def resample(
        self,
        update: bool = True,
        skip: list | tuple = ('icrs', 'itrs', 'time', 'altaz', 'observations'),
        inplace: bool = True,
    ) -> FastRadioBursts:

        copy = self if inplace else self.copy()

        params = (*copy.params, *skip, 'params')

        copy.__dict__ = keyfilter(lambda x: x in params, copy.__dict__)

        copy._energy_normalization

        if update:
            copy.update()

        return copy

    def reduce(self, tolerance: int = 0) -> FastRadioBursts:

        snrs = self.signal_to_noise(total=True)
        snr = xarray.concat(snrs.values(), dim='ALL')
        idx = snr.max('ALL') >= tolerance
        return self[idx.as_numpy()]

    @observation_method
    def teste(self, observation: Observation | None = None):

        return observation.response

    @observation_method
    def altaz(
        self,
        observation: Observation | None = None,
    ) -> coordinates.SkyCoord:

        return getattr(observation, 'altaz', None)

    @observation_method
    def time_delay(self, observation: Observation | None = None):

        return getattr(observation, 'time_delay', None)

    def _peak_density_flux(
        self,
        observation: Observation | None = None,
        channels: int = 1,
    ) -> xarray.DataArray:

        spectral_index = self.spectral_index
        response = observation.get_frequency_response(spectral_index, channels)
        S0 = xarray.DataArray(self._energy_normalization.value, dims='FRB')
        unit = response.attrs['unit'] * self._energy_normalization.unit
        signal = response * S0
        signal.attrs['unit'] = unit.to('Jy')
        return signal

    @observation_method
    def peak_density_flux(
        self,
        observation: Observation | None = None,
        channels: int = 1,
    ) -> xarray.DataArray:

        return self._peak_density_flux(observation, channels)

    def _signal(
        self,
        observation: Observation | None = None,
        channels: int = 1,
    ) -> xarray.DataArray:

        peak_density_flux = self._peak_density_flux(observation, channels)
        in_range = observation.in_range(
            self.redshift, self.low_frequency, self.high_frequency
        )
        signal = peak_density_flux * in_range
        signal.attrs = peak_density_flux.attrs
        return signal

    @observation_method
    def signal(
        self,
        observation: Observation | None = None,
        channels: int = 1,
    ) -> xarray.DataArray:

        return self._signal(observation, channels)

    def _noise(
        self,
        observation: Observation | None = None,
        total: str | bool = False,
        channels: int = 1,
    ) -> xarray.DataArray:

        return observation.get_noise(total, channels)

    @observation_method
    @todense_option
    def noise(
        self,
        observation: Observation | None = None,
        total: str | bool = False,
        channels: int = 1,
        todense: bool = False,
    ) -> xarray.DataArray:

        return self._noise(observation, total, channels)

    def _signal_to_noise(
        self,
        observation: Observation | None = None,
        total: str | bool = False,
        channels: int = 1,
    ) -> xarray.DataArray:

        signal = self._signal(observation, channels)
        noise = self._noise(observation, total, channels)

        return signal / noise

    @observation_method
    @todense_option
    def signal_to_noise(
        self,
        observation: Observation | None = None,
        total: str | bool = False,
        channels: int = 1,
        todense: bool = False,
    ) -> xarray.DataArray:

        return self._signal_to_noise(observation, total, channels)

    @xarrayfy(snr=('SNR',))
    def _triggers(
        self,
        observation: Observation | None = None,
        snr: xarray.DataArray | numpy.ndarray | range = range(1, 11),
        total: str | bool = False,
        channels: int = 1,
    ) -> xarray.DataArray:

        _snr = self._signal_to_noise(observation, total, channels)
        return (_snr >= snr).assign_coords({'SNR': snr.values})

    @observation_method
    @todense_option
    def triggers(
        self,
        observation: Observation | None = None,
        snr: xarray.DataArray | numpy.ndarray | range = range(1, 11),
        total: str | bool = False,
        channels: int = 1,
        todense: bool = False,
    ) -> xarray.DataArray:

        return self._triggers(observation, snr, total, channels)

    @observation_method
    @todense_option
    def counts(
        self,
        observation: Observation | None = None,
        snr: xarray.DataArray | numpy.ndarray | range = range(1, 11),
        total: str | bool = False,
        channels: int = 1,
        todense: bool = True,
    ) -> xarray.DataArray:

        triggers = self._triggers(observation, snr, total, channels)
        return triggers.sum('FRB')

    @default_units('ms', nu='MHz', DM='pc cm^-3')
    def disperse(self, nu, DM):
        return self.disperse_constant * DM / nu**2

    @default_units('', t='ms', w='ms', t0='ms')
    def gaussian(self, t, w, t0=0.0):
        z = (t - t0) / w
        return numpy.exp(-(z**2) / 2)

    @default_units('', t='ms', w='ms', ts='ms', t0='ms')
    def scattered_gaussian(t, w, ts, t0=0.0):

        x = 0.5 * (w / ts) ** 2
        f = numpy.exp(x)

        x = (t0 - t) / ts
        g = numpy.exp(x)

        x = t - t0 - w**2 / ts
        y = w * numpy.sqrt(2)
        h = 1 + erf(x / y)

        ff = f * g * h

        return ff / ff.max(0)

    def _waterfall_noise(
        self,
        observation: Observation | None = None,
        steps: int = 1,
        total: bool = True,
        channels: int = 1,
        kind: str = 'w',
    ) -> xarray.DataArray:

        noise = observation.get_noise(total, channels, True)

        if kind in ('white', 'w'):
            scales = numpy.sqrt(noise)

            waterfall_noise = numpy.stack(
                [numpy.random.normal(scale=scales) for i in range(steps)], -1
            )

            return xarray.DataArray(
                waterfall_noise**2, dims=(*noise.dims, 'TIME')
            )

        elif kind is None:

            return xarray.DataArray(
                sparse.COO(numpy.zeros((channels, steps))),
                dims=(*noise.dims, 'TIME'),
            )
        else:
            raise ValueError('Invalid noise kind.')

    @observation_method
    @todense_option
    def waterfall(
        self,
        observation: Observation | None = None,
        total: bool = True,
        channels: int = 1,
        noise: str = 'w',
        todense: bool = False,
    ) -> xarray.DataArray:

        total_resp = observation.get_response(total=True)

        if isinstance(total_resp.data, sparse.COO):
            not_null = total_resp.data.coords[0]
        else:
            total_resp = sparse.COO(total_resp.data)
            not_null = total_resp.coords[0]

        sub = self[not_null]
        obs = observation[not_null]

        sampling_time = obs.sampling_time
        if hasattr(obs, 'altaz'):
            altaz = obs.altaz
        else:
            altaz = sub.altaz

        peak_time = (altaz.obstime - self.start).to(sampling_time)
        duration = sub.duration.to(sampling_time)

        steps = duration // sampling_time
        steps = steps.value.astype(int).item()

        t = numpy.linspace(0, duration, steps + 1)
        nu = numpy.linspace(*obs.frequency_range, channels + 1)
        nu = (nu[1:] + nu[:-1]) / 2

        dm = sub.dispersion_measure.reshape(-1, 1)
        disp_peak_time = sub.disperse(nu, dm)
        disp_peak_time = peak_time.reshape(-1, 1) + disp_peak_time

        idx = (disp_peak_time[:, [-1, 0]] // sampling_time).astype(int)
        lidx = idx.diff(axis=1).ravel()

        idx[:, 0] = idx[:, 0] - lidx
        idx[:, 1] = idx[:, 1] + lidx
        idx = idx.clip(0, t.size - 1)

        waterfalls = []

        for k in range(sub.size):

            i, j = idx[k]
            t0 = disp_peak_time[k]
            w = sub.pulse_width[k]

            wt = sub.gaussian(t[i:j], w, t0.reshape(-1, 1))
            wt = sparse.COO(wt)
            wt = sparse.pad(wt, ((0, 0), (i, steps + 1 - j)))

            waterfalls.append(wt)

        waterfalls = xarray.DataArray(
            sparse.stack(waterfalls), dims=('FRB', 'CHANNEL', 'TIME')
        )

        signals = sub._signal(obs, channels)
        response = obs.get_response(total=total)

        waterfalls = signals * response * waterfalls
        waterfall = waterfalls.sum('FRB')

        waterfall = waterfall[..., 1:] + waterfall[..., :-1]
        waterfall = waterfall * sampling_time.value / 2
        time = t[:-1] + self.start

        waterfall_noise = self._waterfall_noise(
            obs, steps, total, channels, noise
        )
        waterfall = waterfall + waterfall_noise

        return waterfall.assign_coords(TIME=time.to_datetime())

    def catalog(self, tolerance: int = 1) -> dict:

        catalog = {
            attr: value
            for attr, value in self.__dict__.items()
            if isinstance(value, (numpy.ndarray, units.Quantity))
            and numpy.size(value) == self.size
            and not attr.startswith('_')
            and attr not in self.params
        }

        icrs = self.icrs
        new_args = {'right_ascension': icrs.ra, 'declination': icrs.dec}
        catalog.update(new_args)

        if 'observations' in dir(self):

            observations = {}

            time_delay = self.time_delay()

            if time_delay is not None:
                observations['time_delay'] = time_delay

            if isinstance(self.altaz, types.MethodType):
                altaz = self.altaz()

                if isinstance(altaz, dict):

                    new_args = {
                        coord: {
                            name: getattr(value, coord, None)
                            for name, value in altaz.items()
                        }
                        for coord in ('alt', 'az', 'obstime')
                    }
                    observations.update(new_args)

                elif isinstance(altaz, coordinates.SkyCoord):
                    altaz = {
                        coord: getattr(altaz, coord, None)
                        for coord in ('alt', 'az', 'obstime')
                    }
                    catalog.update(altaz)

            elif isinstance(self.altaz, coordinates.SkyCoord):
                altaz = {
                    coord: getattr(self.altaz, coord, None)
                    for coord in ('alt', 'az', 'obstime')
                }

                catalog.update(altaz)

            snr = self.signal_to_noise(todense=True)
            if isinstance(snr, dict):
                idx = (
                    xarray.concat(
                        [value.max(value.dims[1:]) for value in snr.values()],
                        dim='X',
                    ).max('X')
                    > tolerance
                )
                observations['signal_to_noise'] = snr
            elif isinstance(snr, xarray.DataArray):
                idx = snr.max(snr.dims[1:]) > tolerance
                catalog['signal_to_noise'] = snr

            catalog = valmap(lambda x: x[idx], catalog)
            observations = {
                key: valmap(lambda x: x[idx], value)
                for key, value in observations.items()
            }

            return merge(catalog, observations)

        return catalog

    def save_catalog(self, name: str, tolerance: int = 1):

        catalog = self.catalog(tolerance)
        filename = '{}.cat'.format(name)
        file = bz2.BZ2File(filename, 'wb')
        dill.dump(catalog, file, dill.HIGHEST_PROTOCOL)
        file.close()
