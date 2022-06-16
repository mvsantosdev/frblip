import os
import sys
import warnings

import bz2
import dill

import numpy
import xarray
from sparse import COO

from numpy import random

from operator import itemgetter
from functools import partial, cached_property

from astropy.time import Time
from astropy import units, constants, coordinates
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator

from .random import Redshift, Schechter, SpectralIndex

from .random.dispersion_measure import GalacticDM
from .random.dispersion_measure import InterGalacticDM, HostGalaxyDM

from .observation import Observation, Interferometry
from .cosmology import Cosmology, builtin


def getufunc(method, **kwargs):

    if callable(method):
        return method
    elif hasattr(numpy, method):
        return getattr(numpy, method)
    elif hasattr(numpy.linalg, method):
        func = getattr(numpy.linalg, method)
        return partial(func, **kwargs)


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


class FastRadioBursts(object):

    """Class which defines a Fast Radio Burst population"""

    def __init__(self, size=None, days=1, log_Lstar=44.46, log_L0=41.96,
                 phistar=339, gamma=-1.79, pulse_width=(-6.917, 0.824),
                 zmin=0, zmax=30, ra=(0, 24), dec=(-90, 90), start=None,
                 low_frequency=10.0, high_frequency=10000.0,
                 low_frequency_cal=400.0, high_frequency_cal=1400.0,
                 emission_frame=True, spectral_index='CHIME2021',
                 gal_method='yt2020_analytic', gal_nside=128,
                 host_dist='lognormal', host_source='luo18',
                 host_model=('ALG', 'YMW16'), cosmology='Planck_18',
                 free_electron_bias='Takahashi2021', verbose=True):

        """
        Creates a FRB population object.

        Parameters
        ----------
        size : int
            Number of generated FRBs.
        days : float
            Number of days of observation.
        log_Lstar : float
            log(erg / s)
        log_L_0  : float
            log(erg / s)
        phistar : floar
            Gpc^3 / year
        gamma : float
            Luminosity Schechter index
        wint : (float, float)
            log(ms)
        si : (float, float)
            Spectral Index range
        zmin : float
            Minimum redshift.
        zmax : float
            Maximum redshift.
        ra : (float, float)
            Degree
        dec : (float, float)
            Degree
        start : datetime
            start time
        low_frequency : float
            MHz
        high_frequency : float
            MHz
        gal_method : str
            DM_gal model, default: 'yt2020_analytic'
        gal_nside : int
            DM_gal nside
        cosmology : cosmology model
            default: 'Planck_18'
        free_electron_bias: str
            free electron bias model
            default: 'Takahashi2021'
        verbose : bool
        Returns
        -------
        out: FastRadioBursts object.
        """

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
        """

        Parameters
        ----------
        idx :

        inplace :
            (Default value = False)

        Returns
        -------


        """

        select_dict = {
            name: attr[idx]
            for name, attr in self.__dict__.items()
            if hasattr(attr, 'size')
            and attr.size == self.size
        }

        select_dict['size'] = select_dict['redshift'].size

        observations = getattr(self, 'observations', None)
        if observations is not None:
            observations = {
                name: observation[idx]
                for name, observation in observations.items()
            }
            select_dict['observations'] = observations

        if not inplace:
            out_dict = self.__dict__.copy()
            out_dict.update(select_dict)
            output = FastRadioBursts.__new__(FastRadioBursts)
            output.__dict__.update(out_dict)
            return output

        self.__dict__.update(select_dict)

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
        return (width * units.s).to(units.ms)

    @cached_property
    def emitted_pulse_width(self):
        """ """
        return self.pulse_width / (1 + self.redshift)

    @cached_property
    def itrs_time(self):
        """ """
        time_ms = int(self.duration.to(units.us).value)
        dt = random.randint(time_ms, size=self.size)
        dt = numpy.sort(dt) * units.us
        return self.start + dt

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
        decs = decs.to(units.degree)
        ras = random.uniform(*self.ra_range.value, self.size)
        ras = ras * self.ra_range.unit
        return coordinates.SkyCoord(ras, decs, frame='icrs')

    @cached_property
    def area(self):
        """ """
        x = numpy.sin(self.dec_range).diff().item()
        y = self.ra_range.to(units.rad).diff().item()
        Area = (x * y) * units.rad
        return Area.to(units.degree**2)

    @cached_property
    def __luminosity_distance(self):
        z = self.redshift
        return self.__cosmology.luminosity_distance(z)

    @cached_property
    def __luminosity(self):
        return self.log_luminosity.to(units.erg / units.s)

    @cached_property
    def __flux(self):
        surface = 4 * numpy.pi * self.__luminosity_distance**2
        return (self.__luminosity / surface).to(units.Jy * units.MHz)

    @cached_property
    def __S0(self):
        _sip1 = self.spectral_index + 1
        nu_lp = (self.low_frequency_cal / units.MHz)**_sip1
        nu_hp = (self.high_frequency_cal / units.MHz)**_sip1
        sflux = self.__flux / (nu_hp - nu_lp)
        if self.emission_frame:
            z_factor = (1 + self.redshift)**_sip1
            return sflux * z_factor
        return sflux

    def peak_density_flux(self, frequency):
        si = self.spectral_index.reshape(-1, 1)
        nu_factor = (frequency / units.MHz)**(si + 1)
        nu_factor = nu_factor.diff(axis=-1) / frequency.diff()
        output = self.__S0.reshape(-1, 1) * nu_factor
        return output.squeeze()

    @cached_property
    def itrs(self):
        """ """
        itrs_frame = coordinates.ITRS(obstime=self.itrs_time)
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
        return InterGalacticDM(self.__cosmology)

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

    def obstime(self, location):
        """

        Parameters
        ----------
        location :


        Returns
        -------


        """

        loc = location.get_itrs()
        loc = loc.cartesian.xyz

        path = loc @ self.xyz
        time_delay = path / constants.c

        return self.itrs_time - time_delay

    def altaz(self, location, interp=300):
        """

        Parameters
        ----------
        location :

        interp :
            (Default value = 300)

        Returns
        -------


        """

        obstime = self.obstime(location)
        frame = coordinates.AltAz(location=location, obstime=obstime)
        interp_time = interp * units.s

        with erfa_astrom.set(ErfaAstromInterpolator(interp_time)):
            return self.icrs.transform_to(frame)

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
            if key not in ('icrs', 'itrs', 'itrs_time')
            and numpy.size(value) == self.size
        })

    def __observe(self, telescope, name=None, location=None,
                  altaz=None, sparse=True, dtype=numpy.float64):

        print('Performing observation for telescope {}...'.format(name))

        if 'observations' not in self.__dict__:
            self.observations = {}

        n_obs = len(self.observations)
        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        sampling_time = telescope.sampling_time
        frequency_range = telescope.frequency_range
        width = telescope.frequency_range.diff()

        zmax = self.high_frequency / frequency_range[0] - 1
        zmin = self.low_frequency / frequency_range[-1] - 1
        zmin = zmin.clip(0)

        location = telescope.location if location is None else location
        lon, lat, height = location.lon, location.lat, location.height

        print(
            'Computing positions for {} FRB'.format(self.size),
            'at site lon={:.3f}, lat={:.3f},'.format(lon, lat),
            'height={:.3f}.'.format(height), end='\n\n'
        )

        if altaz is None:
            altaz = self.altaz(location)

        in_range = (zmin <= self.redshift) & (self.redshift <= zmax)
        visible = altaz.alt > 0
        mask = visible & in_range

        vis_frac = round(100 * visible.mean(), 2)
        range_frac = round(100 * in_range.mean(), 2)
        obs_frac = round(100 * mask.mean(), 2)

        print('>>> {}% are visible.'.format(vis_frac))
        print('>>> {}% are in frequency range.'.format(range_frac))
        print('>>> {}% are observable.'.format(obs_frac), end='\n\n')

        dims = 'FRB', obs_name

        resp = telescope.response(altaz[mask])
        shape = self.size, *resp.shape[1:]
        response = numpy.zeros(shape, dtype=dtype)
        response[mask] = resp
        if sparse:
            response = COO(response)
        response = xarray.DataArray(response, dims=dims, name='Response')

        sflux = (self.__S0 / width).to(units.Jy)
        density_flux = xarray.DataArray(sflux.value, dims='FRB', name='Noise')
        density_flux.attrs['unit'] = sflux.unit

        noi = telescope.noise
        noise = xarray.DataArray(noi.value, dims=obs_name, name='Noise')
        noise.attrs['unit'] = noi.unit

        time_delay = telescope.time_array(altaz)
        if time_delay is not None:
            unit = time_delay.unit
            time_delay = xarray.DataArray(time_delay.value, dims=dims,
                                          name='Time Delay')
            time_delay.attrs['unit'] = unit

        observation = Observation(altaz, density_flux, response, noise,
                                  time_delay, frequency_range, sampling_time)

        self.observations[obs_name] = observation

    def observe(self, telescopes, name=None, location=None, altaz=None,
                sparse=False, dtype=numpy.float64, verbose=True):
        """

        Parameters
        ----------
        telescopes :

        location :
            (Default value = None)
        start :
            (Default value = None)
        name :
            (Default value = None)
        altaz :
            (Default value = None)
        verbose :
            (Default value = True)

        Returns
        -------


        """

        old_target = sys.stdout
        sys.stdout = old_target if verbose else open(os.devnull, 'w')

        if type(telescopes) is dict:
            for name, telescope in telescopes.items():
                self.__observe(telescope, name, location, altaz, sparse, dtype)
        else:
            self.__observe(telescopes, name, location, altaz, sparse, dtype)

        sys.stdout = old_target

    def clean(self, names=None):
        """

        Parameters
        ----------
        names :
            (Default value = None)

        Returns
        -------


        """

        if hasattr(self, 'observations'):
            if names is None:
                del self.observations
            elif isinstance(names, str):
                del self.observations[names]
            else:
                for name in names:
                    del self.observations[name]

    def reduce(self, tolerance=0):

        snrs = self.signal_to_noise(total=True)
        snr = xarray.concat(snrs.values(), dim='ALL')
        idx = snr.max('ALL') >= tolerance
        return self[idx.as_numpy()]

    def __time_delay(self, name, channels=1):

        observation = self[name]
        return getattr(observation, 'time_delay', None)

    def __signal(self, name, channels=1):

        observation = self[name]
        spectral_index = self.spectral_index
        return observation.get_response(spectral_index, channels)

    def __noise(self, name, channels=1):

        observation = self[name]
        return observation.get_noise(channels)

    def __signal_to_noise(self, name, channels=1, total=False,
                          method='max', **kwargs):

        func = getufunc(method, **kwargs)

        signal = self.__signal(name, channels)
        noise = self.__noise(name, channels)

        snr = signal / noise

        if isinstance(total, str) and (total in snr.dims):
            snr = snr.reduce(func, dim=total, **kwargs)
        if isinstance(total, list):
            levels = [*filter(lambda x: x in snr.dims, total)]
            snr = snr.reduce(func, dim=levels, **kwargs)
        elif total is True:
            levels = [*filter(lambda x: x not in ('FRB', 'CHANNEL'),
                              snr.dims)]
            snr = snr.reduce(func, dim=levels, **kwargs)

        return snr.squeeze()

    def __triggers(self, name, channels=1, snr=None,
                   total=False, method='max', **kwargs):

        _snr = self.__signal_to_noise(name, channels, total=total,
                                      method=method, **kwargs)
        s = numpy.arange(1, 11) if snr is None else snr
        s = xarray.DataArray(numpy.atleast_1d(s), dims='SNR')
        return (_snr >= s).squeeze()

    def __counts(self, name, channels=1, snr=None, total=False,
                 method='max', **kwargs):

        detected = self.__triggers(name, channels, snr, total,
                                   method, **kwargs)
        return detected.sum('FRB')

    def __count_baselines(self, name, channels=1, snr=None,
                          reference=None, method='max', **kwargs):

        key = 'INTF_{}'.format(name)
        triggers = self.__triggers(key, channels, snr=snr)
        counts = triggers.sum(name)

        if (reference is not None) and isinstance(reference, str):
            key = 'INTF_{}_{}'.format(name, reference)
            triggers = self.__triggers(key, channels, snr=snr, total=reference,
                                       method=method, **kwargs)
            counts += triggers.sum(name)

        return counts

    def __count_over_baselines(self, name, channels=1, snr=None,
                               reference=None, baselines=10,
                               method='max', **kwargs):

        b = xarray.DataArray(numpy.arange(1, baselines+1), dims='Baselines')
        count_baselines = self.__count_baselines(name, channels=channels,
                                                 snr=snr, reference=reference,
                                                 method=method, **kwargs)
        return (count_baselines > b).sum('FRB')

    def __localize(self, name, channels=1, reference='MAIN',
                   trigger=1.5, detect=5, localize=3, base=1,
                   baselines=10, method='sum', **kwargs):

        baselines = xarray.DataArray(numpy.arange(1, baselines+1),
                                     dims='Baselines')

        intf_keys = ['INTF_{}'.format(name),
                     'INTF_{}_{}'.format(name, reference)]
        keys = [name, reference, *intf_keys]

        triggers = self.triggers(keys, channels=channels,
                                 snr=trigger, total=True)
        candidates = numpy.any([
            value for value in triggers.values()
        ], axis=0)
        candidates = xarray.DataArray(candidates, dims='FRB')

        func = getufunc(method, **kwargs)

        snr = self.signal_to_noise(keys, channels, total=True,
                                   method=func, **kwargs)
        snr = func([
            value for value in snr.values()
        ], axis=0)
        snr = xarray.DataArray(snr, dims='FRB')

        detected = (snr > detect) & candidates

        intf_snr = self.signal_to_noise(intf_keys, channels, total=reference,
                                        method=func, **kwargs)
        intf_snr = func([
            value.sum(name) for value in intf_snr.values()
        ], axis=0)
        intf_snr = xarray.DataArray(intf_snr, dims='FRB')

        intf_trig = self.triggers(intf_keys, channels=channels,
                                  snr=base, total=reference)
        intf_trig = sum([
            value.sum(name) for value in intf_trig.values()
        ])

        localized = (intf_snr > localize) & (intf_trig >= baselines)
        localized = detected & localized

        return {
            'candidates': candidates,
            'detected': detected,
            'localized': localized
        }

    def __count_localized(self, name, channels=1, reference='MAIN',
                          trigger=1.5, detect=5, localize=3, base=1,
                          baselines=10, method='sum', **kwargs):

        localized = self.__localize(name, channels, reference, trigger,
                                    detect, localize, base, baselines,
                                    method, **kwargs)

        return {
            key: value.sum('FRB').values
            for key, value in localized.items()
        }

    def __get(self, func_name=None, names=None, channels=1, **kwargs):

        func = self.__getattribute__(func_name)

        if names is None:
            names = self.observations.keys()
        elif isinstance(names, str):
            if names == 'INTF':
                names = self.observations.keys()
                names = [*filter(lambda x: 'INTF' in x, names)]
            elif names == 'AUTO':
                names = self.observations.keys()
                names = [*filter(lambda x: 'INTF' not in x, names)]
            else:
                return func(names, channels, **kwargs)

        output = {
            name: func(name, channels, **kwargs)
            for name in names
        }

        return {
            name: value
            for name, value in output.items()
            if value is not None
        }

    def time_delay(self, names=None):
        """

        Parameters
        ----------
        names :
            (Default value = None)

        Returns
        -------


        """

        return self.__get('_FastRadioBursts__time_delay', names, channels=1)

    def signal(self, names=None, channels=1):
        """

        Parameters
        ----------
        names :
            (Default value = None)
        channels :
            (Default value = False)

        Returns
        -------


        """

        return self.__get('_FastRadioBursts__signal', names, channels)

    def noise(self, names=None, channels=1):
        """

        Parameters
        ----------
        names :
            (Default value = None)
        channels :
            (Default value = False)

        Returns
        -------


        """

        return self.__get('_FastRadioBursts__noise', names, channels)

    def signal_to_noise(self, names=None, channels=1, total=False,
                        method='max', **kwargs):
        """

        Parameters
        ----------
        names :
            (Default value = None)
        channels :
            (Default value = False)
        total :
            (Default value = False)

        Returns
        -------


        """

        return self.__get('_FastRadioBursts__signal_to_noise', names, channels,
                          total=total, method=method, **kwargs)

    def triggers(self, names=None, channels=1, snr=None,
                 total=False, method='max', **kwargs):
        """

        Parameters
        ----------
        names :
            (Default value = None)
        channels :
            (Default value = False)
        total :
            (Default value = False)

        Returns
        -------


        """

        return self.__get('_FastRadioBursts__triggers', names, channels,
                          snr=snr, total=total, method=method, **kwargs)

    def counts(self, names=None, channels=1, snr=None, total=False,
               method='max', **kwargs):
        """

        Parameters
        ----------
        names :
            (Default value = None)
        channels :
            (Default value = False)
        total :
            (Default value = False)

        Returns
        -------


        """

        return self.__get('_FastRadioBursts__counts', names, channels,
                          snr=snr, total=total, method=method, **kwargs)

    def count_baselines(self, names=None, channels=1, snr=None,
                        reference=None, method='max', **kwargs):

        return self.__get('_FastRadioBursts__count_baselines', names,
                          channels, snr=snr, reference=reference,
                          method=method, **kwargs)

    def count_over_baselines(self, names=None, channels=1, snr=None,
                             reference=None, baselines=10, method='max',
                             **kwargs):

        return self.__get('_FastRadioBursts__count_over_baselines', names,
                          channels, snr=snr, reference=reference,
                          baselines=baselines, method=method, **kwargs)

    def localize(self, names=None, channels=1, reference='MAIN', trigger=1.5,
                 detect=5, localize=3, base=1, baselines=10, method='sum',
                 **kwargs):

        return self.__get('_FastRadioBursts__localize', names, channels,
                          reference=reference, trigger=trigger, detect=detect,
                          localize=localize, base=base, baselines=baselines,
                          method=method, **kwargs)

    def count_localized(self, names=None, channels=1, reference='MAIN',
                        trigger=1.5, detect=5, localize=3, base=1,
                        baselines=10, method='sum', **kwargs):

        kw = dict(
            names=names, channels=channels, reference=reference,
            trigger=trigger, detect=detect, localize=localize,
            base=base, baselines=baselines, method=method, **kwargs
        )

        return self.__get('_FastRadioBursts__count_localized', **kw)

    def interferometry(self, namei, namej=None, reference=False,
                       degradation=None, overwrite=False, return_key=False):
        """

        Parameters
        ----------
        namei : str

        namej : str
            (Default value = None)
        time_delay :
            (Default value = True)

        Returns
        -------


        """

        if reference:
            names = [
                name for name in self.observations
                if (name != namei) and ('INTF' not in name)
            ]
            for namej in names:
                self.interferometry(namej)
                self.interferometry(namej, namei)
        else:
            obsi, obsj = self[namei], self[namej]
            if namej is None:
                key = 'INTF_{}'.format(namei)
            else:
                key = 'INTF_{}_{}'.format(namei, namej)

            if (key not in self.observations) or overwrite:
                interferometry = Interferometry(obsi, obsj, degradation)
                self.observations[key] = interferometry
                if return_key:
                    return key
            else:
                warning_message = '{} is already computed. '.format(key) + \
                                  'You may set overwrite=True to recompute.'
                warnings.warn(warning_message)

    def copy(self):
        """ """
        copy = dill.copy(self)
        keys = self.__dict__.keys()

        for key in keys:
            if '_FastRadioBursts__' in key:
                delattr(copy, key)

        return copy

    def save(self, name):
        """
        Parameters
        ----------
        name :


        Returns
        -------
        """
        file_name = '{}.blips'.format(name)
        file = bz2.BZ2File(file_name, 'wb')
        copy = self.copy()
        dill.dump(copy, file, dill.HIGHEST_PROTOCOL)
        file.close()

    @staticmethod
    def load(file):
        """
        Parameters
        ----------
        name :


        Returns
        -------
        """
        file_name = '{}.blips'.format(file)
        file = bz2.BZ2File(file_name, 'rb')
        loaded = dill.load(file)
        file.close()
        return loaded
