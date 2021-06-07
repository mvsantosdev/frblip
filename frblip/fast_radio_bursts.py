import sys

import numpy
import pandas

from numpy import random

from scipy.special import comb
from scipy.integrate import quad, cumtrapz

from functools import partial
from operator import itemgetter
from itertools import combinations, product

from astropy.time import Time
from astropy import cosmology, coordinates, units
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator

from .utils import _DATA, simps, paired_shapes
from .utils import load_params, load_file, sub_dict, xfactors

from .distributions import Redshift, Schechter

from .dispersion import *

from .observation import Observation


class FastRadioBursts(object):

    """
    Class which defines a Fast Radio Burst population
    """

    def __init__(self, n_frb=None, days=1, log_Lstar=44.46, log_L0=41.96,
                 phistar=339, alpha=-1.79, wint=(.13, .33), si=(-15, 15),
                 ra=(0, 24), dec=(-90, 90), zmin=0, zmax=6,
                 cosmo='Planck18_arXiv_v2', lower_frequency=400,
                 higher_frequency=1400, width=True, dispersion=True,
                 verbose=True):

        """
        Creates a FRB population object.

        Parameters
        ----------
        n_frb : int
            Number of generated FRBs.
        days : float
            Number of days of observation.
        zmax : float
            Maximum redshift.
        Lstar : float
            erg / s
        L_0  : float
            erg / s
        phistar : floar
            Gpc^3 / year
        alpha : float
            Luminosity
        wint : (float, float)
            log(ms)
        wint : (float, float)
            Hour
        dec : (float, float)
            Degree
        si : (float, float)
            Spectral Index
        verbose : bool

        Returns
        -------
        out: FastRadioBursts object.

        """

        old_target = sys.stdout
        sys.stdout = open(os.devnull, 'w') if not verbose else old_target

        self._load_params(n_frb, days, log_Lstar, log_L0, phistar,
                          alpha, wint, si, ra, dec, zmin, zmax, cosmo,
                          lower_frequency, higher_frequency)
        self._derived_params()

        self._frb_rate(n_frb, days)

        self._random()
        self._derived_random()

        if dispersion:
            self._dispersion()

        sys.stdout = old_target

    def __len__(self):

        return self.n_frb

    def __getitem__(self, idx):

        if isinstance(idx, str):
            return self.observations[idx]
        if isinstance(idx, slice):
            return self.select(idx, inplace=False)

        idx = numpy.array(idx)
        numeric = numpy.issubdtype(idx.dtype, numpy.signedinteger)
        boolean = numpy.issubdtype(idx.dtype, numpy.bool_)
        if numeric or boolean or islice:
            return self.select(idx, inplace=False)
        if numpy.issubdtype(idx.dtype, numpy.str_):
            return itemgetter(*idx)(self.observations)
        return None

    def iterfrbs(self, start=0, stop=None, step=1):

        stop = self.n_frb if stop is None else stop

        for i in range(start, stop, step):
            yield self[i]

    def iterchunks(self, size=1, start=0, stop=None, retindex=False):

        stop = self.n_frb if stop is None else stop

        if retindex:
            for i in range(start, stop, size):
                j = i + size
                yield i, j, self[i:j]
        else:
            for i in range(start, stop, size):
                j = i + size
                yield self[i:j]

    def select(self, idx, inplace=False):

        out_dict = self.to_dict()
        n_frb = out_dict['n_frb']

        out_dict.update({
            k: v[idx] for k, v in out_dict.items()
            if numpy.ndim(v) and 'noise' not in k
            and 'frequency_bands' not in k
            and itemgetter(0)(v.shape) == n_frb
        })

        out_dict['n_frb'] = out_dict['redshift'].size

        if not inplace:
            return FastRadioBursts.from_dict(out_dict)

        self.__dict__.update(out_dict)

    def observation_time(self, start_time=None):

        if start_time is None:
            start_time = Time(Time.now().mjd, format='mjd')

        return start_time + self.time.ravel()

    def altaz(self, location, start_time=None, interp=300):

        obstime = self.observation_time(start_time)

        frame = coordinates.AltAz(location=location,
                                  obstime=obstime)

        interp_time = interp * units.s

        with erfa_astrom.set(ErfaAstromInterpolator(interp_time)):

            altaz = self.sky_coord.transform_to(frame)

        return altaz

    def density_flux(self, nu):

        sip = 1 + self.spectral_index[numpy.newaxis]
        nup = nu.value[:, numpy.newaxis]**sip
        flux = self.__S0 * numpy.diff(nup, axis=0)
        diff_nu = numpy.diff(nu)

        return flux.T / diff_nu

    def _frb_rate(self, n_frb, days):

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

            self.area = self._sky_area()
            sky_fraction = self.area / units.astrophys.sp
            self.rate = self.__sky_rate * sky_fraction.to(1)

        else:

            self.area = units.astrophys.sp
            self.rate = self.__sky_rate

        self.rate = int(self.rate.value) * self.rate.unit

        print('FRB rate =', self.rate)

        if n_frb is None:

            self.n_frb = int(self.rate.value * days)
            self.duration = days * (24 * units.hour)

        else:

            self.n_frb = n_frb
            self.duration = (n_frb / self.rate).to(units.hour)

        print(self.n_frb, 'FRBs will be simulated, the actual rate is',
              self.rate, '.\nTherefore it corrensponds to', self.duration,
              'of observation. \n')

    def _load_params(self, n_frb, days, log_Lstar, log_L0, phistar,
                     alpha, wint, si, ra, dec, zmin, zmax, cosmo,
                     lower_frequency, higher_frequency):

        self.zmin = zmin
        self.zmax = zmax
        self.log_L0 = log_L0 * units.LogUnit(units.erg / units.s)
        self.log_Lstar = log_Lstar * units.LogUnit(units.erg / units.s)
        self.phistar = phistar / (units.Gpc**3 * units.year)
        self.alpha = alpha

        self.w_mean, self.w_std = wint
        self.si_min, self.si_max = si

        self.ra_range = numpy.array(ra) * units.hourangle
        self.dec_range = numpy.array(dec) * units.degree

        self.lower_frequency = lower_frequency * units.MHz
        self.higher_frequency = higher_frequency * units.MHz

        self.cosmology = cosmo

    def _derived_params(self):

        self.__cosmology = cosmology.__dict__.get(self.cosmology)

        self.__xmin = self.log_L0 - self.log_Lstar
        self.__zdist = Redshift(zmin=self.zmin, zmax=self.zmax,
                                cosmology=self.__cosmology)
        xmin = (self.log_L0 - self.log_Lstar).to(1).value
        self.__lumdist = Schechter(xmin, self.alpha)

        Lum = self.phistar / self.__lumdist.pdf_norm
        Vol = 1 / self.__zdist.pdf_norm

        self.__sky_rate = (Lum * Vol).to(1 / units.day)

    def _random(self):

        self.redshift = self.__zdist.rvs(size=self.n_frb)
        loglum = self.__lumdist.log_rvs(size=self.n_frb)
        self.log_luminosity = loglum * units.LogUnit() + self.log_Lstar
        self.pulse_width = random.lognormal(self.w_mean, self.w_std,
                                            size=self.n_frb) * units.ms

        time_ms = int(self.duration.to(units.ms).value)
        self.time = random.randint(time_ms, size=self.n_frb) * units.ms

        self.spectral_index = random.uniform(self.si_min, self.si_max,
                                             self.n_frb)

        self._coordinates()

    def _derived_random(self):

        z = self.redshift
        lumdist = self.__cosmology.luminosity_distance(z)
        self.__luminosity_distance = lumdist

        surface = 4 * numpy.pi * self.__luminosity_distance**2

        self.__luminosity = self.log_luminosity.to(units.erg / units.s)
        self.__flux = (self.__luminosity / surface).to(units.Jy * units.MHz)

        self.__arrived_pulse_width = (1 + self.redshift) * self.pulse_width

        _sip1 = self.spectral_index + 1

        nu_lp = self.lower_frequency.value**_sip1
        nu_hp = self.higher_frequency.value**_sip1

        self.__S0 = self.__flux / (nu_hp - nu_lp)
        self.__S0 = self.__S0

    def _sky_area(self):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        x = numpy.sin(self.dec_range).diff() * units.rad
        y = self.ra_range.to(units.rad).diff()

        Area = x * y

        return Area[0].to(units.degree**2)

    def _coordinates(self):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        sin = numpy.sin(self.dec_range)

        args = random.uniform(*sin, self.n_frb)

        decs = (numpy.arcsin(args) * units.rad).to(units.degree)
        ras = random.uniform(*self.ra_range.value,
                             self.n_frb) * self.ra_range.unit

        self.sky_coord = coordinates.SkyCoord(ras, decs, frame='icrs')

    def _dispersion(self):

        _zp1 = 1 + self.redshift

        lon = self.sky_coord.galactic.l
        lat = self.sky_coord.galactic.b

        gal_DM = galactic_dispersion(lon, lat)
        host_DM = host_galaxy_dispersion(self.redshift)
        src_DM = source_dispersion(self.n_frb)
        igm_DM = igm_dispersion(self.redshift, zmax=self.zmax)

        egal_DM = igm_DM + (src_DM + host_DM) / _zp1

        self.__galactic_dispersion = gal_DM
        self.__host_dispersion = host_DM
        self.__source_dispersion = src_DM
        self.__igm_dispersion = igm_DM

        self.__extra_galactic_dispersion = egal_DM

        self.__dispersion = gal_DM + egal_DM

    def _observe(self, Telescope, location=None, start=None,
                 name=None, altaz=None):

        if 'observations' not in self.__dict__:
            self.observations = {}

        noise = Telescope.noise()

        location = Telescope.location if location is None else location
        start_time = Telescope.start_time if start is None else start

        frequency_bands = Telescope.frequency_bands
        sampling_time = Telescope.sampling_time

        altaz = self.altaz(location, start) if altaz is None else altaz
        response = Telescope.response(altaz)

        obs = Observation(response, noise, frequency_bands,
                          sampling_time, altaz)

        n_obs = len(self.observations)
        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        self.observations[obs_name] = obs

    def observe(self, Telescopes, location=None, start=None,
                name=None, altaz=None):

        if type(Telescopes) is dict:
            for Name, Telescope in Telescopes.items():
                self._observe(Telescope, location, start, Name, altaz)
        else:
            self._observe(Telescopes, location, start, name, altaz)

    def clear(self, names=None):

        if names is None:
            del self.observations
        elif isinstance(names, str):
            del self.observations[names]
        else:
            for name in names:
                del self.observations[name]

    def clear_xcorr(self, names=None):
        del self.__xcorr

    def _signal(self, name, channels=False):

        obs = self.observations[name]
        response = obs.response
        ndim = tuple(range(1, response.ndim))

        if channels:
            signal = self.density_flux(obs.frequency_bands)
            signal = numpy.expand_dims(signal, axis=ndim)
            return response[..., numpy.newaxis] * signal

        signal = self.density_flux(obs.frequency_bands[[0, -1]])
        return response * signal

    def signal(self, channels=False):

        return {
            obs: self._signal(obs, channels)
            for obs in self.observations
        }

    def _noise(self, name, channels=False):

        obs = self.observations[name]
        if channels:
            return obs.noise
        inoise = (1 / obs.noise**2).sum(-1)
        return 1 / numpy.sqrt(inoise)

    def noise(self, channels=False):

        return {
            obs: self._noise(obs, channels)
            for obs in self.observations
        }

    def _signal_to_noise(self, name, channels=False):

        signal = self._signal(name, channels)
        noise = self._noise(name, channels)

        return (signal / noise).value

    def signal_to_noise(self, names=None, channels=False):

        if names is None:
            observations = self.observations
        elif isinstance(names, str):
            return self._signal_to_noise(names, channels)
        else:
            values = itemgetter(*names)(self.observations)
            observations = dict(zip(names, values))

        return {
            obs: self._signal_to_noise(obs, channels)
            for obs in observations
        }

    def cross_correlation(self, namei, namej):

        freqi = self.observations[namei].frequency_bands
        freqj = self.observations[namei].frequency_bands

        freq = numpy.row_stack((freqi, freqj))
        freq = numpy.unique(freq, axis=0).ravel()

        respi = self.observations[namei].response[..., numpy.newaxis]
        respj = self.observations[namej].response[:, numpy.newaxis]
        response = numpy.sqrt(respi * respj)

        noisei = self.observations[namei].noise[:, numpy.newaxis].to(units.Jy)
        noisej = self.observations[namej].noise[numpy.newaxis].to(units.Jy)
        noise = numpy.sqrt(noisei * noisej)

        observation = Observation(response=response, noise=noise,
                                  frequency_bands=freq)

        if "_FastRadioBursts__xcorr" not in self.__dict__:
            self.__xcorr = {}

        self.__xcorr[(namei, namej)] = observation

    def interferometry(self, **telescopes):

        names = [*telescopes.keys()]
        n_scopes = len(names)

        freqs = numpy.stack([
            self.observations.get(name).frequency_bands
            for name in names
        ])

        frequency = numpy.unique(freqs, axis=0).ravel()

        observations = [self.observations[name] for name in names]

        shapes = [
            observation._Observation__n_beam
            for observation in observations
        ]

        shape = numpy.concatenate(shapes[:n_scopes])
        shapes = paired_shapes(shape)

        xnames = [*combinations(names, 2)]

        for xname in xnames:
            if xname not in self.observations:
                self.cross_correlation(*xname)

        observations += [self.__xcorr[name] for name in xnames]

        responses = numpy.broadcast_arrays(*[
            observation.response.reshape(-1, *shape)
            for shape, observation in zip(shapes, observations)
        ])

        noises = numpy.broadcast_arrays(*[
            observation.noise.reshape(*shape, -1).to(units.Jy)
            for shape, observation in zip(shapes, observations)
        ])

        responses = numpy.stack(responses, axis=-1)
        noises = numpy.stack(noises, axis=-1)

        array = partial(numpy.array, ndmin=1)
        counts = map(array, telescopes.values())
        counts = numpy.row_stack([*product(*counts)])
        factors = numpy.apply_along_axis(xfactors, 1, counts)

        for count, factor in zip(counts, factors):

            response = (factor * responses / 2).sum(-1)
            inoise = (2 * factor / noises**2).sum(-1)
            noise = units.Jy / numpy.sqrt(inoise)

            observation = Observation(response=response, noise=noise,
                                      frequency_bands=frequency)

            keys = ['{}x{}'.format(c, l) for c, l in zip(count, names)]
            key = 'INTF_{}'.format('_'.join(keys)).replace('_1x', '_')

            self.observations[key] = observation

    def split_beams(self, name, key='BEAM'):

        observation = self.observations.pop(name)
        splitted = observation.split_beams()

        self.observations.update({
            '{}_{}_{}'.format(name, key, beam): obs
            for beam, obs in enumerate(splitted)
        })

    def to_dict(self):

        out_dict = {
            key: name
            for key, name in self.__dict__.items()
            if '_FastRadioBursts__' not in key
        }

        sky_coord = out_dict.pop('sky_coord')

        out_dict['ra'] = sky_coord.ra
        out_dict['dec'] = sky_coord.dec

        if 'observations' in out_dict:

            observations = out_dict.pop('observations')

            keys = [*observations.keys()]

            out_dict['observations'] = numpy.array([
                key for key in keys if 'INTF' not in key
            ])

            for name in out_dict['observations']:

                flag = '{}__'.format(name)
                observation = observations[name].to_dict(flag)

                out_dict.update(observation)

        out_dict.update({
            'u_{}'.format(key): value.unit.to_string()
            for key, value in out_dict.items()
            if hasattr(value, 'unit')
        })

        out_dict.update({
            key: value.value
            for key, value in out_dict.items()
            if hasattr(value, 'unit')
        })

        return out_dict

    def save(self, file, compressed=True):

        out_dict = self.to_dict()

        if compressed:
            numpy.savez_compressed(file, **out_dict)
        else:
            numpy.savez(file, **out_dict)

    @staticmethod
    def from_dict(input_dict):

        params = load_params(input_dict)
        output = FastRadioBursts.__new__(FastRadioBursts)

        if 'observations' in input_dict.keys():

            observations = params.pop('observations')
            output.observations = {}

            for name in observations:

                flag = '{}__'.format(name)
                obs = sub_dict(params, flag=flag, pop=True)
                observation = Observation.from_dict(obs)
                output.observations[name] = observation

        ra = params.pop('ra')
        dec = params.pop('dec')

        sky_coords = coordinates.SkyCoord(ra, dec, frame='icrs')
        params['sky_coord'] = sky_coords

        output.__dict__.update(params)
        output._derived_params()
        output._derived_random()

        return output

    @staticmethod
    def load(file):
        input_dict = load_file(file)
        return FastRadioBursts.from_dict(input_dict)
