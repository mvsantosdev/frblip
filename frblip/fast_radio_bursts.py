import sys

import numpy

from numpy import random

import pandas

from scipy.special import comb

from itertools import combinations, product

from scipy.integrate import quad, cumtrapz

from astropy.time import Time

from astropy import cosmology, coordinates, units

from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator

from .utils import load_params, sub_dict
from .utils import _all_sky_area, _DATA, schechter, rvs_from_cdf, simps

from .dispersion import *

from .observation import Observation


class FastRadioBursts(object):

    """
    Class which defines a Fast Radio Burst population
    """

    def __init__(self, n_frb=None, days=1, Lstar=2.9e44, L_0=9.1e41,
                 phistar=339, alpha=-1.79, wint=(.13, .33), si=(-15, 15),
                 ra=(0, 24), dec=(-90, 90), zmax=6, cosmo=None, verbose=True,
                 lower_frequency=400, higher_frequency=1400, dispersion=True,
                 width=True):

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

        self._load_params(n_frb, days, Lstar, L_0, phistar,
                          alpha, wint, si, ra, dec, zmax, cosmo,
                          lower_frequency, higher_frequency)

        self._coordinates()

        self._primary()
        self._derived()

        if dispersion:
            self._dispersion()

        sys.stdout = old_target

    def to_csv(self, file, **kwargs):

        df = self.to_pandas()

        df.to_csv(file, **kwargs)

    def to_pandas(self):

        df = pandas.DataFrame({
            'Redshift': self.redshift.ravel(),
            'Comoving Distance (Mpc)': self.comoving_distance.ravel(),
            'Luminosity (erg/s)': self.luminosity.ravel(),
            'Flux (Jy * MHz)': self.flux.ravel(),
            'Right Ascension (hour)': self.sky_coord.ra.to(units.hourangle),
            'Declination (degree)': self.sky_coord.dec,
            'Galactic Longitude (degree)': self.sky_coord.galactic.l,
            'Galactic Latitude (degree)': self.sky_coord.galactic.b,
            'Intrinsic Pulse Width (ms)': self.pulse_width.ravel(),
            'Arrived Pulse Width (ms)': self.arrived_pulse_width.ravel(),
            'Galactic DM (pc/cm^3)': self.galactic_dispersion.ravel(),
            'Host Galaxy DM (pc/cm^3)': self.host_dispersion.ravel(),
            'Intergalactic Medium DM (pc/cm^3)': self.igm_dispersion.ravel(),
            'Source DM (pc/cm^3)': self.source_dispersion.ravel(),
            'Dispersion Measure (pc/cm^3)': self.dispersion.ravel(),
            'Spectral Index': self.spectral_index.ravel(),
            'Time (ms)': self.time.ravel()
        }).sort_values('Time (ms)', ignore_index=True)

        return df

    def __len__(self):

        return self.n_frb

    def __getitem__(self, idx):

        return self.select(idx, inplace=False)

    def select(self, idx, inplace=False):

        frbs = self if inplace else FastRadioBursts(n_frb=0, verbose=False)

        update_dict = {
            'sky_coord': self.sky_coord[idx],
            'dec_range': self.dec_range,
            'ra_range': self.ra_range
        }

        keys = [*update_dict.keys()]

        update_dict.update({
            key: value[idx] if numpy.size(value) > 1 else value
            for key, value in self.__dict__.items()
            if key not in keys
        })

        frbs.__dict__.update(update_dict)

        frbs.n_frb = frbs.redshift.size

        if 'observations' in self.__dict__.keys():

            frbs.observations = {
                name: obs[idx]
                for name, obs in self.observations.items()
            }

        if not inplace:

            return frbs

    def observation_time(self, start_time=None):

        if start_time is None:
            start_time = Time.now()

        return start_time + self.time.ravel()

    def altaz(self, location, start_time=None, interp=300):

        obstime = self.observation_time(start_time)

        frame = coordinates.AltAz(location=location,
                                  obstime=obstime)

        interp_time = interp * units.s

        with erfa_astrom.set(ErfaAstromInterpolator(interp_time)):

            altaz = self.sky_coord.transform_to(frame)

        return altaz

    def specific_flux(self, nu):

        nut = nu.value.reshape(-1, 1)

        S = self.__S0 * (nut**self.spectral_index)

        return S.T

    def _frb_rate(self, n_frb, days):

        print("Computing the FRB rate ...")

        self.sky_rate = self._sky_rate()

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
            self.rate = self.sky_rate * (self.area / _all_sky_area).to(1)

        else:

            self.area = _all_sky_area
            self.rate = self.sky_rate

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

    def _load_params(self, n_frb, days, Lstar, L_0, phistar,
                     alpha, wint, si, ra, dec, zmax, cosmo,
                     lower_frequency, higher_frequency):

        self.zmax = zmax
        self.L_0 = L_0 * units.erg / units.s
        self.Lstar = Lstar * units.erg / units.s
        self.phistar = phistar / (units.Gpc**3 * units.year)
        self.alpha = alpha

        self.w_mean, self.w_std = wint
        self.si_min, self.si_max = si

        self.ra_range = numpy.array(ra) * units.hourangle
        self.dec_range = numpy.array(dec) * units.degree

        self.lower_frequency = lower_frequency * units.MHz
        self.higher_frequency = higher_frequency * units.MHz

        self.cosmo = cosmology.Planck18_arXiv_v2 if cosmo is None else cosmo

        self._frb_rate(n_frb, days)

    def _primary(self):

        z, co = self._z_dist()

        self.redshift = z
        self.comoving_distance = co

        self.luminosity = self._luminosity()
        self.pulse_width = random.lognormal(self.w_mean, self.w_std,
                                            size=self.n_frb) * units.ms

        time_ms = int(self.duration.to(units.ms).value)
        self.time = random.randint(time_ms, size=self.n_frb) * units.ms

        self.spectral_index = random.uniform(self.si_min,
                                             self.si_max,
                                             self.n_frb)

    def _derived(self):

        _zp1 = 1 + self.redshift

        self.__luminosity_distance = _zp1 * self.comoving_distance

        surface = 4 * numpy.pi * self.__luminosity_distance**2

        self.__flux = (self.luminosity / surface).to(units.Jy * units.MHz)

        self.__arrived_pulse_width = _zp1 * self.pulse_width

        _sip1 = self.spectral_index + 1

        nu_low = self.lower_frequency.value**_sip1
        nu_high = self.higher_frequency.value**_sip1

        self.__S0 = _sip1 * self.__flux / (nu_high - nu_low)
        self.__S0 = self.__S0 / units.MHz

    def _sky_area(self):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        x = numpy.sin(self.dec_range).diff() * units.rad
        y = self.ra_range.to(units.rad).diff()

        Area = x * y

        return Area[0].to(units.degree**2)

    def _sky_rate(self):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        r = self.L_0 / self.Lstar

        Lum, eps = self.phistar * quad(schechter, r, numpy.inf,
                                       args=(self.alpha,))

        Vol = self.cosmo.comoving_volume(self.zmax)

        return (Lum * Vol).to(1/units.day)

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

    def _z_dist(self):

        U = random.random(self.n_frb)

        zs = numpy.linspace(.0, self.zmax, 100)

        cdfz = self.cosmo.comoving_volume(zs).value
        codists = self.cosmo.comoving_distance(zs)

        zout = numpy.interp(x=U, xp=cdfz / cdfz[-1], fp=zs)
        rout = numpy.interp(x=zout, xp=zs, fp=codists)

        return zout, rout

    def _luminosity(self):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        xmin = (self.L_0 / self.Lstar).value

        rvs = rvs_from_cdf(schechter, xmin, 3,
                           size=self.n_frb,
                           alpha=self.alpha)

        return self.Lstar * rvs

    def _observe(self, Telescope, location=None, start=None,
                 name=None, altaz=None, full=True):

        if 'observations' not in self.__dict__.keys():

            self.observations = {}

        noise = Telescope.noise()

        location = Telescope.location if location is None else location
        start_time = Telescope.start_time if start is None else start

        frequency_bands = Telescope.frequency_bands
        sampling_time = Telescope.sampling_time

        if full:

            altaz = self.altaz(location, start) if altaz is None else altaz
            response = Telescope.response(altaz)

        else:

            noise = noise.min(0)
            response = numpy.array([[1.0]])

            obstime = self.observation_time(start)

            altaz = coordinates.AltAz(location=location, obstime=obstime)

        obs = Observation(response, noise, frequency_bands,
                          sampling_time, altaz)

        n_obs = len(self.observations)
        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        self.observations[obs_name] = obs

    def observe(self, Telescopes, location=None, start=None,
                name=None, altaz=None, full=True):

        if type(Telescopes) is dict:

            for Name, Telescope in Telescopes.items():

                self._observe(Telescope, location, start,
                              Name, altaz, full)

        else:

            self._observe(Telescopes, location, start,
                          name, altaz, full)

    def clear(self, names=None):

        if names is None:

            del self.observations

        else:

            for name in list(names):

                del self.observations[name]

    def _signal(self, name, channels=False):

        obs = self.observations[name]

        signal = self.specific_flux(obs._Observation__frequency)
        signal = simps(signal)

        nshape = len(obs.response.shape) - 1
        ishape = numpy.ones(nshape, dtype=int)

        response = obs.response[..., numpy.newaxis]
        signal = signal.reshape((self.n_frb, *ishape,
                                 obs._Observation__n_channel))

        signal = response * signal

        if channels:

            return signal

        return numpy.average(signal, weights=obs._Observation__band_widths,
                             axis=-1)

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

    def _signal_to_noise(self, observation, channels=False):

        signal = self._signal(observation, channels)
        noise = self._noise(observation, channels)

        return (signal / noise).value

    def signal_to_noise(self, channels=False):

        return {
            obs: self._signal_to_noise(obs, channels)
            for obs in self.observations
        }

    def interferometry(self, **telescopes):

        observations = self.observations

        names = numpy.array([*telescopes.keys()])

        n_scopes = len(names)
        n_frb = self.n_frb

        n_channel = numpy.unique([
                observations[name]._Observation__n_channel
                for name in names
            ]).item(0)

        n_beam = numpy.concatenate([
                observations[name]._Observation__n_beam
                for name in names
            ])

        shapes = numpy.diag(n_beam - 1)
        shapes = shapes + numpy.ones_like(shapes)

        responses = [
            0.5 * observations[name].response.reshape((n_frb, *shape))
            for name, shape in zip(names, shapes)
        ]

        responses += [
            numpy.sqrt(ri * rj)
            for ri, rj in combinations(responses, 2)
        ]

        inoises_sq = [
            2 / observations[name].noise.reshape((*shape, n_channel))**2
            for name, shape in zip(names, shapes)
        ]

        inoises_sq += [
            numpy.sqrt(ni * nj)
            for ni, nj in combinations(inoises_sq, 2)
        ]

        telescopes.update({
            name: [count]
            for name, count in telescopes.items()
            if type(count) is int
        })

        for counts in product(*telescopes.values()):

            arr = numpy.array(counts)

            factors = (arr * (arr - 1)) // 2
            xfactors = numpy.array([ci * cj for ci, cj
                                    in combinations(arr, 2)])
            factors = numpy.concatenate((factors, xfactors))

            response = [
                factor * resp
                for factor, resp in zip(factors, responses)
            ]

            inoise_sq = [
                factor * in_sq
                for factor, in_sq in zip(factors, inoises_sq)
            ]

            nunit = numpy.unique([q.unit for q in inoise_sq]).item(0)

            response = sum(response, numpy.zeros(()))
            inoise_sq = sum(inoise_sq, numpy.zeros(()) * nunit)

            noise = 1 / numpy.sqrt(inoise_sq)

            frequency_bands = observations[names[0]].frequency_bands

            obs = Observation(response=response, noise=noise,
                              frequency_bands=frequency_bands)

            obs.response = numpy.squeeze(obs.response)
            obs.noise = numpy.squeeze(obs.noise)

            if obs.response.ndim == 1:
                obs.response = obs.response.reshape(-1, 1)
                obs.noise = obs.noise.reshape(1, -1)

            keys = ['{}x{}'.format(c, l) for c, l in zip(counts, names)]
            key = 'INTF_{}'.format('_'.join(keys).replace('1x', ''))

            self.observations[key] = obs

    def split_beams(self, name, key='BEAM'):

        observation = self.observations.pop(name)
        splitted = observation.split_beams()

        self.observations.update({
            '{}_{}_{}'.format(name, key, beam): obs
            for beam, obs in enumerate(splitted)
        })

    def save(self, file):

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

            out_dict['observations'] = numpy.array([*observations.keys()])

            for name in out_dict['observations']:

                flag = '{}__'.format(name)
                observation = observations[name].to_dict(flag)

                out_dict.update(observation)

        out_dict.update({
            'u_{}'.format(key): value.unit.to_string()
            for key, value in out_dict.items()
            if hasattr(value, 'unit')
        })

        numpy.savez(file, **out_dict)

    @staticmethod
    def load(file):

        input_dict = load_params(file)
        output = FastRadioBursts(n_frb=0, verbose=False)

        if 'observations' in input_dict.keys():

            observations = input_dict.pop('observations')
            output.observations = {}

            for name in observations:

                obs = sub_dict(input_dict, flag='{}__'.format(name))
                observation = Observation.from_dict(obs)
                output.observations[name] = observation

        ra = input_dict.pop('ra')
        dec = input_dict.pop('dec')
        sky_coords = coordinates.SkyCoord(ra, dec, frame='icrs')
        input_dict['sky_coord'] = sky_coords

        output.__dict__.update(input_dict)
        output._derived()

        return output
