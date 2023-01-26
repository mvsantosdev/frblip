import os
import sys

import bz2
import dill

import numpy
import xarray
import sparse

from numpy import random

from operator import itemgetter
from functools import cached_property
from toolz.dicttoolz import merge, valmap, valfilter

from astropy.time import Time
from astropy import units, coordinates, constants

from .random import Redshift, Schechter, SpectralIndex

from .random.dispersion_measure import GalacticDM
from .random.dispersion_measure import InterGalacticDM, HostGalaxyDM

from .cosmology import Cosmology

from .basic_sampler import BasicSampler
from .decorators import observation_method, todense_option, default_units


class FastRadioBursts(BasicSampler):
    """ """

    DISP_CONST = constants.e.emu**2 * constants.c
    DISP_CONST = DISP_CONST / (2 * numpy.pi * constants.m_e)

    def __init__(self,
                 size: int = None,
                 duration: float = 1,
                 log_Lstar: float = 44.46,
                 log_L0: float = 41.96,
                 phistar: float = 339,
                 gamma: float = -1.79,
                 pulse_width: (float, float) = (-6.917, 0.824),
                 zmin: float = 0.0,
                 zmax: float = 30.0,
                 ra_range: (float, float) = (0, 24),
                 dec_range: (float, float) = (-90, 90),
                 start: 'Time or None' = None,
                 low_frequency: float = 10.0,
                 high_frequency: float = 10000.0,
                 low_frequency_cal: float = 400.0,
                 high_frequency_cal: float = 1400.0,
                 emission_frame: bool = False,
                 spectral_index: str = 'CHIME2021',
                 gal_method: str = 'yt2020_analytic',
                 gal_nside: int = 128,
                 host_dist: str = 'lognormal',
                 host_source: str = 'luo18',
                 host_model: (str, str) = ('ALG', 'YMW16'),
                 cosmology: str = 'Planck_18',
                 igm_model: str = 'Takahashi2021',
                 free_electron_bias: str = 'Takahashi2021',
                 verbose: bool = True):

        """

        Parameters
        ----------
        size : int or None
             Number of generated FRB.
             (Default value = None)
        days : int or None
             (Default value = 1)
        log_Lstar :
             (Default value = 44.46)
        log_L0 :
             (Default value = 41.96)
        phistar :
             (Default value = 339.0)
        gamma :
             (Default value = -1.79)
        pulse_width :
             (Default value = (-6.917, 0.824))
        zmin :
             (Default value = 0)
        zmax :
             (Default value = 30)
        ra :
             (Default value = (0, 24))
        dec :
             (Default value = (-90, 90))
        start :
             (Default value = None)
        low_frequency :
             (Default value = 10.0)
        high_frequency :
             (Default value = 10000.0)
        low_frequency_cal :
             (Default value = 400.0)
        high_frequency_cal :
             (Default value = 1400.0)
        emission_frame :
             (Default value = False)
        spectral_index :
             (Default value = 'CHIME2021')
        gal_method :
             (Default value = 'yt2020_analytic')
        gal_nside :
             (Default value = 128)
        host_dist :
             (Default value = 'lognormal')
        host_source :
             (Default value = 'luo18')
        host_model :
             (Default value = ('ALG', 'YMW16'))
        cosmology :
             (Default value = 'Planck_18')
        free_electron_bias :
             (Default value = 'Takahashi2021')
        verbose :
             (Default value = True)

        Returns
        -------

        """

        old_target = sys.stdout
        sys.stdout = old_target if verbose else open(os.devnull, 'w')

        self._load_params(duration, log_Lstar, log_L0, phistar, gamma,
                          pulse_width, zmin, zmax, ra_range, dec_range, start,
                          low_frequency, high_frequency,
                          low_frequency_cal, high_frequency_cal,
                          emission_frame, spectral_index, gal_method,
                          gal_nside, host_dist, host_source, host_model,
                          cosmology, igm_model, free_electron_bias)

        self._frb_rate(size)
        self.__S0
        self.kind = 'FRB'

        sys.stdout = old_target

    @default_units(duration='day', log_L0='dex(erg / s)',
                   log_Lstar='dex(erg / s)', phistar='1 / (Gpc^3 yr)',
                   ra_range='hourangle', dec_range='deg',
                   low_frequency='MHz', high_frequency='MHz',
                   low_frequency_cal='MHz', high_frequency_cal='MHz')
    def _load_params(self, duration, log_Lstar, log_L0, phistar, gamma,
                     pulse_width, zmin, zmax, ra_range, dec_range, start,
                     low_frequency, high_frequency, low_frequency_cal,
                     high_frequency_cal, emission_frame, spectral_index,
                     gal_method, gal_nside, host_dist, host_source, host_model,
                     cosmology, igm_model, free_electron_bias):

        self.duration = duration
        self.zmin = zmin
        self.zmax = zmax
        self.log_L0 = log_L0
        self.log_Lstar = log_Lstar
        self.phistar = phistar
        self.gamma = gamma
        self.w_mean, self.w_std = pulse_width
        self.__spec_idx_dist = SpectralIndex(spectral_index)
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

    def __len__(self):
        return self.size

    def __getitem__(self,
                    idx: [
                        str, slice, numpy.signedinteger,
                        numpy.bool_, numpy.str_
                    ]):

        if isinstance(idx, str):
            return self.observations[idx]
        if isinstance(idx, slice):
            return self.select(idx, inplace=False)

        idx = numpy.array(idx)
        numeric = numpy.issubdtype(idx.dtype, numpy.signedinteger)
        boolean = numpy.issubdtype(idx.dtype, numpy.bool_)
        if numeric or boolean:
            return self.select(idx, inplace=False)
        if numpy.issubdtype(idx.dtype, numpy.str_):
            return itemgetter(*idx)(self.observations)
        return None

    def select(self,
               idx: [str, slice, numpy.signedinteger, numpy.bool_, numpy.str_],
               inplace: bool = False):
        """

        Parameters
        ----------
        idx :

        inplace :
             (Default value = False)

        Returns
        -------

        """

        if not inplace:
            mock = self.copy(clear=False)
            mock.select(idx, inplace=True)
            return mock

        self.__dict__.update({
            name: attr[idx]
            for name, attr in self.__dict__.items()
            if hasattr(attr, 'size')
            and numpy.size(attr) == self.size
        })

        if hasattr(self, 'observations'):
            self.observations.update({
                name: observation[idx]
                for name, observation in self.observations.items()
            })

        self.size = self.redshift.size

    def iterfrbs(self, start=0, stop=None, step=1):
        """

        Parameters
        ----------
        start :
             (Default value = 0)
        stop :
             (Default value = None)
        step :
             (Default value = 1)

        Returns
        -------

        """

        stop = self.size if stop is None else stop
        for i in range(start, stop, step):
            yield self[i]

    def iterchunks(self, size=1, start=0, stop=None, retindex=False):
        """

        Parameters
        ----------
        size :
             (Default value = 1)
        start :
             (Default value = 0)
        stop :
             (Default value = None)
        retindex :
             (Default value = False)

        Returns
        -------

        """

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
    def __cosmology(self):
        kw = {
            'name': self.cosmology,
            'free_electron_bias': self.free_electron_bias
        }
        return Cosmology(**kw)

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
        return Schechter(self.__xmin, self.gamma)

    @cached_property
    def sky_rate(self):
        Lum = self.phistar / self.__lumdist.pdf_norm
        Vol = 1 / self.__zdist.pdf_norm
        return (Lum * Vol).to('1 / day')

    @cached_property
    def redshift(self):
        """ """
        return self.__zdist.rvs(size=self.size)

    @cached_property
    def log_luminosity(self):
        """ """
        loglum = self.__lumdist.log_rvs(size=self.size)
        return loglum * units.LogUnit() + self.log_Lstar

    @cached_property
    def pulse_width(self):
        """ """
        width = random.lognormal(self.w_mean, self.w_std, size=self.size)
        return (width * units.s).to('ms')

    @cached_property
    def emitted_pulse_width(self):
        """ """
        return self.pulse_width / (1 + self.redshift)

    @cached_property
    def time(self):
        """ """
        dt = random.random(size=self.size) * self.duration
        return self.start + numpy.sort(dt)

    @cached_property
    def spectral_index(self):
        """ """
        return self.__spec_idx_dist.rvs(self.size)

    @cached_property
    def icrs(self):
        """ """

        sin = numpy.sin(self.dec_range)
        args = random.uniform(*sin, self.size)
        decs = numpy.arcsin(args) * units.rad
        decs = decs.to('deg')
        ras = random.uniform(*self.ra_range.value, self.size)
        ras = ras * self.ra_range.unit
        return coordinates.SkyCoord(ras, decs, frame='icrs')

    @cached_property
    def area(self):
        """ """

        x = numpy.sin(self.dec_range).diff().item()
        y = self.ra_range.to('rad').diff().item()
        Area = (x * y) * units.rad
        return Area.to('deg^2')

    @cached_property
    def luminosity_distance(self):
        """ """
        z = self.redshift
        return self.__cosmology.luminosity_distance(z)

    @cached_property
    def __luminosity(self):
        return self.log_luminosity.to('erg / s')

    @cached_property
    def flux(self):
        """ """
        surface = 4 * numpy.pi * self.luminosity_distance**2
        return (self.__luminosity / surface).to('Jy MHz')

    @cached_property
    def __S0(self):
        _sip1 = self.spectral_index + 1
        nu_lp = (self.low_frequency_cal / units.MHz)**_sip1
        nu_hp = (self.high_frequency_cal / units.MHz)**_sip1
        sflux = self.flux / (nu_hp - nu_lp)
        if self.emission_frame:
            z_factor = (1 + self.redshift)**_sip1
            return sflux * z_factor
        return sflux

    @cached_property
    def itrs(self):
        """ """
        itrs_frame = coordinates.ITRS(obstime=self.time)
        return self.icrs.transform_to(itrs_frame)

    @property
    def xyz(self):
        """ """
        return self.itrs.cartesian.xyz

    @property
    def galactic(self):
        """ """
        return self.icrs.galactic

    @cached_property
    def __gal_dm(self):
        return GalacticDM(self.gal_nside, self.gal_method)

    @cached_property
    def __igm_dm(self):
        return InterGalacticDM(free_electron_model=self.igm_model,
                               cosmology=self.__cosmology)

    @cached_property
    def __host_dm(self):
        return HostGalaxyDM(self.host_source, self.host_model,
                            self.__cosmology, self.host_dist)

    @cached_property
    def galactic_dm(self):
        """ """
        gl = self.galactic.l
        gb = self.galactic.b
        return self.__gal_dm(gl, gb)

    @cached_property
    def igm_dm(self):
        """ """
        z = self.redshift
        return self.__igm_dm(z)

    @cached_property
    def host_dm(self):
        """ """

        z = self.redshift
        return self.__host_dm(z)

    @cached_property
    def extra_galactic_dm(self):
        """ """

        z = self.redshift
        igm = self.igm_dm
        host = self.host_dm
        return igm + host / (1 + z)

    @cached_property
    def dispersion_measure(self):
        """ """
        return self.galactic_dm + self.extra_galactic_dm

    def _frb_rate(self, size):

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

            size = (rate * self.duration).to(1).value
            self.size = int(size)
            self.rate = rate
        else:
            raise TypeError("size must be an integer or None.")
        self.rate = self.rate.to('1 / day')

    def update(self):
        """ """

        self.time = self.time + self.duration
        kw = {
            'x': self.itrs.x,
            'y': self.itrs.y,
            'z': self.itrs.z,
            'obstime': self.time
        }
        self.itrs = coordinates.ITRS(**kw)

        if isinstance(self.altaz, coordinates.SkyCoord):
            kw = {
                'alt': self.altaz.alt,
                'az': self.altaz.az,
                'obstime': self.altaz.obstime + self.duration
            }
            self.altaz = coordinates.AltAz(**kw)
        elif 'observations' in dir(self):
            for name in self.observations:
                self.observations[name].update(self.duration)

    def shuffle(self, update=True):
        """

        Parameters
        ----------
        update :
             (Default value = True)

        Returns
        -------

        """

        idx = numpy.arange(self.size)
        numpy.random.shuffle(idx)

        self.__dict__.update({
            key: value[idx]
            for key, value in self.__dict__.items()
            if key not in ('icrs', 'itrs', 'itrs_time', 'altaz')
            and numpy.size(value) == self.size
        })

        if update:
            self.update()

    def reduce(self, tolerance=0):
        """

        Parameters
        ----------
        tolerance :
             (Default value = 0)

        Returns
        -------

        """

        snrs = self.signal_to_noise(total=True)
        snr = xarray.concat(snrs.values(), dim='ALL')
        idx = snr.max('ALL') >= tolerance
        return self[idx.as_numpy()]

    @observation_method
    def teste(self, observation):
        """

        Parameters
        ----------
        name :


        Returns
        -------

        """

        return observation.response

    @observation_method
    def altaz(self, observation):
        """

        Parameters
        ----------
        names :


        Returns
        -------

        """

        return getattr(observation, 'altaz', None)

    @observation_method
    def time_delay(self, observation):
        """

        Parameters
        ----------
        names :


        Returns
        -------

        """

        return getattr(observation, 'time_delay', None)

    def _peak_density_flux(self, observation, channels=1):
        """

        Parameters
        ----------
        names :

        channels :
             (Default value = 1)

        Returns
        -------

        """

        spectral_index = self.spectral_index
        response = observation.get_frequency_response(spectral_index, channels)
        S0 = xarray.DataArray(self.__S0.value, dims='FRB')
        unit = response.attrs['unit'] * self.__S0.unit
        signal = response * S0
        signal.attrs['unit'] = unit.to('Jy')
        return signal

    @observation_method
    def peak_density_flux(self, observation, channels=1):

        return self._peak_density_flux(observation, channels)

    def _signal(self, observation, channels=1):

        peak_density_flux = self._peak_density_flux(observation, channels)
        in_range = observation.in_range(self.redshift, self.low_frequency,
                                        self.high_frequency)
        signal = peak_density_flux * in_range
        signal.attrs = peak_density_flux.attrs
        return signal

    @observation_method
    def signal(self, observation, channels=1):

        """

        Parameters
        ----------
        names :

        channels :
             (Default value = 1)

        Returns
        -------

        """

        return self._signal(observation, channels)

    def _noise(self, observation, total=False, channels=1):

        return observation.get_noise(total, channels)

    @observation_method
    @todense_option()
    def noise(self, observation, total=False, channels=1):

        """

        Parameters
        ----------
        names :

        total :
             (Default value = False)
        channels :
             (Default value = 1)

        Returns
        -------

        """

        return self._noise(observation, total, channels)

    def _signal_to_noise(self, observation, total=False, channels=1):

        signal = self._signal(observation, channels)
        noise = self._noise(observation, total, channels)

        return signal / noise

    @observation_method
    @todense_option(False)
    def signal_to_noise(self, observation, total=False, channels=1):

        """

        Parameters
        ----------
        names :

        total :
             (Default value = False)
        channels :
             (Default value = 1)

        Returns
        -------

        """

        return self._signal_to_noise(observation, total, channels)

    def _triggers(self, observation, snr=None, total=False, channels=1):

        _snr = self._signal_to_noise(observation, total, channels)
        s = numpy.arange(1, 11) if snr is None else snr
        s = xarray.DataArray(numpy.atleast_1d(s), dims='SNR')
        return (_snr >= s).squeeze()

    @observation_method
    @todense_option(False)
    def triggers(self, observation, snr=None, total=False, channels=1):

        """

        Parameters
        ----------
        names :

        snr :
             (Default value = None)
        total :
             (Default value = False)
        channels :
             (Default value = 1)

        Returns
        -------

        """

        return self._triggers(observation, snr, total, channels)

    @observation_method
    @todense_option()
    def counts(self, observation, channels=1, snr=None, total=False):
        """

        Parameters
        ----------
        names :

        channels :
             (Default value = 1)
        snr :
             (Default value = None)
        total :
             (Default value = False)

        Returns
        -------

        """

        triggers = self._triggers(observation, snr, total, channels)
        return triggers.sum('FRB')

    def disperse(self, nu, DM):
        return (self.DISP_CONST * DM / nu**2).to('ms')

    def gaussian(self, t, w, t0=0.0):
        z = (t - t0) / w
        return numpy.exp(- z**2 / 2)

    def _waterfall_noise(self, observation, steps=1, total=True, channels=1,
                         kind='w'):

        noise = observation.get_noise(total, channels, True)

        if kind in ('white', 'w'):
            scales = numpy.sqrt(noise)

            waterfall_noise = numpy.stack([
                numpy.random.normal(scale=scales)
                for i in range(steps)
            ], -1)

            return xarray.DataArray(
                waterfall_noise**2,
                dims=(*noise.dims, 'TIME')
            )

        elif kind is None:

            return xarray.DataArray(
                sparse.COO(numpy.zeros((channels, steps))),
                dims=(*noise.dims, 'TIME')
            )
        else:
            raise ValueError('Invalid noise kind.')

    @observation_method
    @todense_option
    def waterfall(self, observation, total=True, channels=1, noise='w'):

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
        idx = idx.clip(0, t.size-1)

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
            sparse.stack(waterfalls),
            dims=('FRB', 'CHANNEL', 'TIME')
        )

        signals = sub._signal(obs, channels)
        response = obs.get_response(total=total)

        waterfalls = signals * response * waterfalls
        waterfall = waterfalls.sum('FRB')

        waterfall = waterfall[..., 1:] + waterfall[..., :-1]
        waterfall = waterfall * sampling_time.value / 2
        time = t[:-1] + self.start

        waterfall_noise = self._waterfall_noise(obs, steps, total,
                                                channels, noise)
        waterfall = waterfall + waterfall_noise

        return waterfall.assign_coords(TIME=time.to_datetime())

    def catalog(self, tolerance=1):
        """

        Parameters
        ----------
        tolerance :
             (Default value = 1)

        Returns
        -------

        """

        catalog = {
            attr: value
            for attr, value in self.__dict__.items()
            if numpy.size(value) == self.size
            and '_FastRadioBursts__' not in attr
        }

        icrs = catalog.pop('icrs')
        catalog.update({
            'right_ascension': icrs.ra,
            'declination': icrs.dec
        })

        if 'altaz' in catalog:
            altaz = catalog.pop('altaz')
            catalog.update({
                'altitude': altaz.alt,
                'azimuth': altaz.az,
                'obstime': altaz.obstime
            })

        observations = {}

        if 'observations' in dir(self):
            observations.update({
                'signal_to_noise': self.signal_to_noise(),
                'time_delay': self.time_delay()
            })
            if 'altaz' not in dir(self):
                altaz = self.altaz()
                observations.update({
                    coord: {
                        name: getattr(value, coord)
                        for name, value in altaz.items()
                    }
                    for coord in ('alt', 'az', 'obstime')
                })
            observations = valfilter(lambda x: x is not None,
                                     observations)

        catalog = valfilter(lambda x: x is not None, catalog)

        if (tolerance > 0) and ('signal_to_noise' in observations):
            snr = observations['signal_to_noise']
            idx = xarray.concat([
                value.max(value.dims[1:])
                for value in snr.values()
            ], dim='X').max('X') > tolerance

            catalog = valmap(lambda x: x[idx], catalog)
            observations = {
                key: valmap(lambda x: x[idx], value)
                for key, value in observations.items()
            }

        return merge(catalog, observations)

    def save_catalog(self, name, tolerance=1):
        """

        Parameters
        ----------
        name :

        tolerance :
             (Default value = 1)

        Returns
        -------

        """

        catalog = self.catalog(tolerance)
        filename = '{}.cat'.format(name)
        file = bz2.BZ2File(filename, 'wb')
        dill.dump(catalog, file, dill.HIGHEST_PROTOCOL)
        file.close()
