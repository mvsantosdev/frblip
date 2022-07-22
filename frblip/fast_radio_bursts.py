import os
import sys

import bz2
import dill

import numpy
import xarray

from numpy import random

from operator import itemgetter
from functools import cached_property
from toolz.dicttoolz import merge, valmap, valfilter

from astropy.time import Time
from astropy import units, coordinates

from .random import Redshift, Schechter, SpectralIndex

from .random.dispersion_measure import GalacticDM
from .random.dispersion_measure import InterGalacticDM, HostGalaxyDM

from .cosmology import Cosmology, builtin

from .basic_sampler import BasicSampler


def blips(size=None, days=1, log_Lstar=44.46, log_L0=41.96,
          phistar=339, gamma=-1.79, pulse_width=(-6.917, 0.824),
          zmin=0, zmax=30, ra=(0, 24), dec=(-90, 90), start=None,
          low_frequency=10.0, high_frequency=10000.0,
          low_frequency_cal=400.0, high_frequency_cal=1400.0,
          emission_frame=False, spectral_index='CHIME2021',
          gal_method='yt2020_analytic', gal_nside=128,
          host_dist='lognormal', host_source='luo18',
          host_model=('ALG', 'YMW16'), cosmology='Planck_18',
          free_electron_bias='Takahashi2021', verbose=True):

    return FastRadioBursts(size, days, log_Lstar, log_L0, phistar, gamma,
                           pulse_width, zmin, zmax, ra, dec, start,
                           low_frequency, high_frequency, low_frequency_cal,
                           high_frequency_cal, emission_frame, spectral_index,
                           gal_method, gal_nside, host_dist, host_source,
                           host_model, cosmology, free_electron_bias, verbose)


def load(file):
    return FastRadioBursts.load(file)


def load_catalog(name):

    file = bz2.BZ2File(name, 'rb')
    catalog = dill.load(file)
    file.close()
    return catalog


class FastRadioBursts(BasicSampler):

    def __init__(self, size=None, days=1, log_Lstar=44.46, log_L0=41.96,
                 phistar=339, gamma=-1.79, pulse_width=(-6.917, 0.824),
                 zmin=0, zmax=30, ra=(0, 24), dec=(-90, 90), start=None,
                 low_frequency=10.0, high_frequency=10000.0,
                 low_frequency_cal=400.0, high_frequency_cal=1400.0,
                 emission_frame=False, spectral_index='CHIME2021',
                 gal_method='yt2020_analytic', gal_nside=128,
                 host_dist='lognormal', host_source='luo18',
                 host_model=('ALG', 'YMW16'), cosmology='Planck_18',
                 free_electron_bias='Takahashi2021', verbose=True):

        old_target = sys.stdout
        sys.stdout = old_target if verbose else open(os.devnull, 'w')

        self.__load_params(size, days, log_Lstar, log_L0, phistar, gamma,
                           pulse_width, zmin, zmax, ra, dec, start,
                           low_frequency, high_frequency,
                           low_frequency_cal, high_frequency_cal,
                           emission_frame, spectral_index, gal_method,
                           gal_nside, host_dist, host_source, host_model,
                           cosmology, free_electron_bias)
        self.__frb_rate(size, days)
        self.__S0
        self.kind = 'FRB'

        sys.stdout = old_target

    def __load_params(self, size, days, log_Lstar, log_L0, phistar, gamma,
                      pulse_width, zmin, zmax, ra, dec, start, low_frequency,
                      high_frequency, low_frequency_cal, high_frequency_cal,
                      emission_frame, spectral_index, gal_method, gal_nside,
                      host_dist, host_source, host_model,
                      cosmology, free_electron_bias):

        self.zmin = zmin
        self.zmax = zmax
        self.log_L0 = log_L0 * units.LogUnit(units.erg / units.s)
        self.log_Lstar = log_Lstar * units.LogUnit(units.erg / units.s)
        self.phistar = phistar / (units.Gpc**3 * units.year)
        self.gamma = gamma
        self.w_mean, self.w_std = pulse_width
        self.__spec_idx_dist = SpectralIndex(spectral_index)
        self.ra_range = numpy.array(ra) * units.hourangle
        self.dec_range = numpy.array(dec) * units.degree
        self.low_frequency = low_frequency * units.MHz
        self.high_frequency = high_frequency * units.MHz
        self.low_frequency_cal = low_frequency_cal * units.MHz
        self.high_frequency_cal = high_frequency_cal * units.MHz
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

    def __getitem__(self, idx):

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

    def select(self, idx, inplace=False):

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

        stop = self.size if stop is None else stop
        for i in range(start, stop, step):
            yield self[i]

    def iterchunks(self, size=1, start=0, stop=None, retindex=False):

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
        params = builtin[self.cosmology]
        return Cosmology(**params, free_electron_bias=self.free_electron_bias)

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
    def __sky_rate(self):
        Lum = self.phistar / self.__lumdist.pdf_norm
        Vol = 1 / self.__zdist.pdf_norm
        return (Lum * Vol).to(1 / units.day)

    @cached_property
    def redshift(self):
        return self.__zdist.rvs(size=self.size)

    @cached_property
    def log_luminosity(self):
        loglum = self.__lumdist.log_rvs(size=self.size)
        return loglum * units.LogUnit() + self.log_Lstar

    @cached_property
    def pulse_width(self):
        width = random.lognormal(self.w_mean, self.w_std, size=self.size)
        return (width * units.s).to(units.ms)

    @cached_property
    def emitted_pulse_width(self):
        return self.pulse_width / (1 + self.redshift)

    @cached_property
    def itrs_time(self):
        time_ms = int(self.duration.to(units.us).value)
        dt = random.randint(time_ms, size=self.size)
        dt = numpy.sort(dt) * units.us
        return self.start + dt

    @cached_property
    def spectral_index(self):
        return self.__spec_idx_dist.rvs(self.size)

    @cached_property
    def icrs(self):

        sin = numpy.sin(self.dec_range)
        args = random.uniform(*sin, self.size)
        decs = numpy.arcsin(args) * units.rad
        decs = decs.to(units.degree)
        ras = random.uniform(*self.ra_range.value, self.size)
        ras = ras * self.ra_range.unit
        return coordinates.SkyCoord(ras, decs, frame='icrs')

    @cached_property
    def area(self):

        x = numpy.sin(self.dec_range).diff().item()
        y = self.ra_range.to(units.rad).diff().item()
        Area = (x * y) * units.rad
        return Area.to(units.degree**2)

    @cached_property
    def luminosity_distance(self):
        z = self.redshift
        return self.__cosmology.luminosity_distance(z)

    @cached_property
    def __luminosity(self):
        return self.log_luminosity.to(units.erg / units.s)

    @cached_property
    def flux(self):
        surface = 4 * numpy.pi * self.luminosity_distance**2
        return (self.__luminosity / surface).to(units.Jy * units.MHz)

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
        itrs_frame = coordinates.ITRS(obstime=self.itrs_time)
        return self.icrs.transform_to(itrs_frame)

    @property
    def xyz(self):
        return self.itrs.cartesian.xyz

    @property
    def galactic(self):
        return self.icrs.galactic

    @cached_property
    def __gal_dm(self):
        return GalacticDM(self.gal_nside, self.gal_method)

    @cached_property
    def __igm_dm(self):
        return InterGalacticDM(self.__cosmology)

    @cached_property
    def __host_dm(self):
        return HostGalaxyDM(self.host_source, self.host_model,
                            self.__cosmology, self.host_dist)

    @cached_property
    def galactic_dm(self):
        gl = self.galactic.l
        gb = self.galactic.b
        return self.__gal_dm(gl, gb)

    @cached_property
    def igm_dm(self):
        z = self.redshift
        return self.__igm_dm(z)

    @cached_property
    def host_dm(self):

        z = self.redshift
        return self.__host_dm(z)

    @cached_property
    def extra_galactic_dm(self):

        z = self.redshift
        igm = self.igm_dm
        host = self.host_dm
        return igm + host / (1 + z)

    @cached_property
    def dispersion_measure(self):
        return self.galactic_dm + self.extra_galactic_dm

    def __frb_rate(self, size, days):

        print("Computing the FRB rate ...")

        all_ra = self.ra_range != numpy.array([0, 360]) * units.degree
        all_dec = self.dec_range != numpy.array([-90, 90]) * units.degree

        if all_ra.all() or all_dec.all():
            print(
                'The FoV is restricted between',
                '{} < ra < {} and {} < dec < {}.'.format(*self.ra_range,
                                                         *self.dec_range),
                '\nMake sure that the survey is also',
                'restricted to this region.'
            )
            sky_fraction = self.area / units.spat
            self.rate = self.__sky_rate * sky_fraction.to(1)
        else:
            self.rate = self.__sky_rate
        self.rate = int(self.rate.value) * self.rate.unit

        print('FRB rate =', self.rate)

        if size is None:
            self.size = int(self.rate.value * days)
            self.duration = days * (24 * units.hour)
        else:
            self.size = size
            self.duration = (size / self.rate).to(units.hour)

        print(self.size, 'FRBs will be simulated, the actual rate is',
              self.rate, '.\nTherefore it corrensponds to', self.duration,
              'of observation. \n')

    def shuffle(self):

        idx = numpy.arange(self.size)
        numpy.random.shuffle(idx)

        self.__dict__.update({
            key: value[idx]
            for key, value in self.__dict__.items()
            if key not in ('icrs', 'itrs', 'itrs_time', 'altaz')
            and numpy.size(value) == self.size
        })

    def reduce(self, tolerance=0):

        snrs = self.signal_to_noise(total=True)
        snr = xarray.concat(snrs.values(), dim='ALL')
        idx = snr.max('ALL') >= tolerance
        return self[idx.as_numpy()]

    def _altaz(self, name):

        observation = self[name]
        return getattr(observation, 'altaz', None)

    def _time_delay(self, name):

        observation = self[name]
        return getattr(observation, 'time_delay', None)

    def _peak_density_flux(self, name, channels=1):

        observation = self[name]
        spectral_index = self.spectral_index
        response = observation.get_frequency_response(spectral_index, channels)
        S0 = xarray.DataArray(self.__S0.value, dims='FRB')
        unit = response.attrs['unit'] * self.__S0.unit
        signal = response * S0
        signal.attrs['unit'] = unit.to(units.Jy)
        return signal

    def _signal(self, name, channels=1):

        observation = self[name]
        peak_density_flux = self._peak_density_flux(name, channels)
        in_range = observation.in_range(self.redshift, self.low_frequency,
                                        self.high_frequency)
        signal = peak_density_flux * in_range
        signal.attrs = peak_density_flux.attrs
        return signal

    def _noise(self, name, total=False, channels=1):

        observation = self[name]
        return observation.get_noise(total, channels)

    def _signal_to_noise(self, name, total=False, channels=1):

        signal = self._signal(name, channels)
        noise = self._noise(name, total, channels)

        return signal / noise

    def _triggers(self, name, snr=None, total=False, channels=1):

        _snr = self._signal_to_noise(name, total, channels)
        s = numpy.arange(1, 11) if snr is None else snr
        s = xarray.DataArray(numpy.atleast_1d(s), dims='SNR')
        return (_snr >= s).squeeze()

    def _counts(self, name, channels=1, snr=None, total=False):

        triggers = self._triggers(name, snr, total, channels)
        return triggers.sum('FRB')

    def catalog(self, tolerance=1):

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
                'azimuth': altaz.az
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
                    for coord in ('alt', 'az')
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

        catalog = self.catalog(tolerance)
        filename = '{}.cat'.format(name)
        file = bz2.BZ2File(filename, 'wb')
        dill.dump(catalog, file, dill.HIGHEST_PROTOCOL)
        file.close()
