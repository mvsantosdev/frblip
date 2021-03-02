import sys

import numpy

from numpy import random

import pandas

from scipy.special import comb

from itertools import combinations

from scipy.integrate import quad, cumtrapz

from astropy.time import Time

from astropy import cosmology, coordinates, units

from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator


from .utils import _all_sky_area, _DATA, schechter, rvs_from_cdf, simps
from .utils import null_coordinates, null_location, null_obstime, super_zip

from .dispersion import *

from .observation import Observation


class FastRadioBursts(object):

    """
    Class which defines a Fast Radio Burst population
    """

    def __init__(self, n_frb=None, days=1, Lstar=2.9e44, L_0=9.1e41,
                 phistar=339, alpha=-1.79, wint=(.13, .33), si=(-15, 15),
                 ra=(0, 24), dec=(-90, 90), zmax=6, cosmo=None, verbose=True,
                 lower_frequency=400, higher_frequency=1400):

        """
        Creates a FRB population object.

        Parameters
        ----------
        ngen : int
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
        out: FRB object.

        """

        old_target = sys.stdout
        sys.stdout = open(os.devnull, 'w') if not verbose else old_target

        self._load_params(n_frb, days, Lstar, L_0, phistar,
                          alpha, wint, si, ra, dec, zmax, cosmo,
                          lower_frequency, higher_frequency)

        self._coordinates()

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

        frbs = self if inplace else FastRadioBursts(verbose=False)

        frbs.redshift = self.redshift[idx]
        frbs.comoving_distance = self.comoving_distance[idx]
        frbs.luminosity = self.luminosity[idx]
        frbs.flux = self.flux[idx]
        frbs.sky_coord = self.sky_coord[idx]
        frbs.pulse_width = self.pulse_width[idx]
        frbs.arrived_pulse_width = self.arrived_pulse_width[idx]
        frbs.galactic_dispersion = self.galactic_dispersion[idx]
        frbs.host_dispersion = self.host_dispersion[idx]
        frbs.igm_dispersion = self.igm_dispersion[idx]
        frbs.source_dispersion = self.source_dispersion[idx]
        frbs.dispersion = self.dispersion[idx]
        frbs.spectral_index = self.spectral_index[idx]
        frbs.time = self.time[idx]
        frbs.S0 = self.S0[idx]

        if 'observations' in self.__dict__.keys():

            frbs.observations = {
                name: obs[idx]
                for name, obs in self.observations.items()
            }

        frbs.n_frb = len(frbs.redshift)

        if not inplace:

            frbs.zmax = self.zmax
            frbs.L_0 = self.L_0
            frbs.Lstar = self.Lstar
            frbs.phistar = self.phistar
            frbs.alpha = self.alpha

            frbs.ra_range = self.ra_range
            frbs.dec_range = self.dec_range

            frbs.lower_frequency = self.lower_frequency
            frbs.higher_frequency = self.higher_frequency

            frbs.area = self.area
            frbs.rate = self.rate
            frbs.sky_rate = self.sky_rate

            frbs.duration = self.duration

            return frbs

    def observation_time(self, start_time=None):

        if start_time is None:
            start_time = Time.now()

        return start_time + self.time.ravel()

    def get_local_coordinates(self, location, start_time=None):

        obstime = self.observation_time(start_time)

        AltAzFrame = coordinates.AltAz(location=location,
                                       obstime=obstime)

        with erfa_astrom.set(ErfaAstromInterpolator(300 * units.s)):

            AltAzCoords = self.sky_coord.transform_to(AltAzFrame)

        return AltAzCoords

    def specific_flux(self, nu):

        return self.S0 * (nu.value**self.spectral_index)

    def peak_density_flux(self, survey):

        _sip1 = self.spectral_index + 1

        num = survey.frequency_bands.value**_sip1
        num = numpy.diff(num, axis=0)

        Speak = self.flux * num / (self.dnu * survey.band_widths)

        return Speak.T

    def _frb_rate(self, n_frb, days):

        print("Computing the FRB rate ...")

        self.sky_rate = self._sky_rate()

        all_ra = self.ra_range != numpy.array([0, 24]) * units.hourangle
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

        self.ra_range = numpy.array(ra) * units.hourangle
        self.dec_range = numpy.array(dec) * units.degree

        self.lower_frequency = lower_frequency * units.MHz
        self.higher_frequency = higher_frequency * units.MHz

        self.cosmo = cosmology.Planck18_arXiv_v2 if cosmo is None else cosmo

        self._frb_rate(n_frb, days)

        z, co = self._z_dist()

        self.redshift = z
        self.comoving_distance = co

        _zp1 = 1 + z

        self.luminosity_distance = _zp1 * co

        self.luminosity = self._luminosity()

        surface = 4 * numpy.pi * self.luminosity_distance**2

        self.flux = (self.luminosity / surface).to(units.Jy * units.MHz)

        self.pulse_width = random.lognormal(*wint, (self.n_frb, 1)) * units.ms

        self.arrived_pulse_width = _zp1 * self.pulse_width

        time_ms = int(self.duration.to(units.ms).value)

        self.time = random.randint(time_ms, size=(self.n_frb, 1)) * units.ms

        self.spectral_index = random.uniform(*si, (self.n_frb, 1))

        _sip1 = self.spectral_index + 1

        nu_low = lower_frequency**_sip1
        nu_high = higher_frequency**_sip1

        self.S0 = _sip1 * self.flux / (nu_high - nu_low)
        self.S0 = self.S0 / units.MHz

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

        lon = self.sky_coord.galactic.l.reshape(-1, 1)
        lat = self.sky_coord.galactic.b.reshape(-1, 1)

        gal_DM = galactic_dispersion(lon, lat)
        host_DM = host_galaxy_dispersion(self.redshift)
        src_DM = source_dispersion((self.n_frb, 1))
        igm_DM = igm_dispersion(self.redshift, zmax=self.zmax)

        egal_DM = igm_DM + (src_DM + host_DM) / _zp1

        self.galactic_dispersion = gal_DM
        self.host_dispersion = host_DM
        self.source_dispersion = src_DM
        self.igm_dispersion = igm_DM

        self.extra_galactic_dispersion = egal_DM

        self.dispersion = gal_DM + egal_DM

    def _z_dist(self):

        U = random.random((self.n_frb, 1))

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
                           size=(self.n_frb, 1),
                           alpha=self.alpha)

        return self.Lstar * rvs

    def _observe(self, Telescope, location=None, start_time=None,
                 name=None, local_coordinates=None, full=True):

        if 'observations' not in self.__dict__.keys():

            self.observations = {}

        noise = Telescope.minimum_temperature / Telescope.gain.reshape(-1, 1)
        noise = noise.to(units.Jy)

        location = Telescope.location if location is None else location
        start_time = Telescope.start_time if start_time is None else start_time

        frequency_bands = Telescope.frequency_bands
        sampling_time = Telescope.sampling_time

        if full:

            if local_coordinates is None:

                local_coordinates = self.get_local_coordinates(location,
                                                               start_time)

            response = Telescope.selection(local_coordinates)

        else:

            response = numpy.array([[1.0]])
            noise = noise.min(-1)

            obstime = self.observation_time(start_time)

            local_coordinates = null_coordinates(az=numpy.nan,
                                                 alt=numpy.nan,
                                                 obstime=obstime,
                                                 location=location)

        obs = Observation(response, noise, frequency_bands,
                          sampling_time, local_coordinates,
                          full)

        n_obs = len(self.observations)
        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        self.observations[obs_name] = obs

    def observe(self, Telescopes, location=None, start_time=None,
                name=None, local_coordinates=None, full=True):

        if type(Telescopes) is dict:

            for Name, Telescope in Telescopes.items():

                self._observe(Telescope, location, start_time,
                              Name, local_coordinates, full)

        else:

            self._observe(Telescopes, location, start_time,
                          name, local_coordinates, full)

    def clear(self, names=None):

        if names is None:

            del self.observations

        else:

            for name in list(names):

                del self.observations[name]

    def _signal(self, name, channels=False):

        obs = self.observations[name]

        signal = self.specific_flux(obs._frequency)
        signal = simps(signal)

        nshape = len(obs.response.shape) - 1
        ishape = numpy.ones(nshape, dtype=int)

        response = obs.response[..., numpy.newaxis]
        signal = signal.reshape((self.n_frb, *ishape,
                                 obs.n_channel))

        signal = response * signal

        if channels:

            return signal

        return numpy.average(signal, weights=obs._band_widths, axis=-1)

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
                observations[name].n_channel
                for name in names
            ]).item(0)

        n_beam = numpy.concatenate([
                observations[name].n_beam
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

        iterator = super_zip(*telescopes.values())

        for counts in iterator:

            factors = (counts * (counts - 1)) // 2
            xfactors = numpy.array([ci * cj for ci, cj
                                    in combinations(counts, 2)])
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

            response = sum(response, numpy.empty(()))
            inoise_sq = sum(inoise_sq, numpy.empty(()) * nunit)

            noise = 1 / numpy.sqrt(inoise_sq)

            frequency_bands = observations[names[0]].frequency_bands

            obs = Observation(response=response, noise=noise,
                              frequency_bands=frequency_bands,
                              full=False)

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
            'L_0': self.L_0,
            'Lstar': self.Lstar,
            'phistar': self.phistar,
            'ra_range': self.ra_range,
            'dec_range': self.dec_range,
            'lower_frequency': self.lower_frequency,
            'higher_frequency': self.higher_frequency,
            'sky_rate': self.sky_rate,
            'area': self.area,
            'rate': self.rate,
            'duration': self.duration,
            'redshift': self.redshift,
            'comoving_distance': self.comoving_distance,
            'luminosity': self.luminosity,
            'ra': self.sky_coord.ra,
            'dec': self.sky_coord.dec,
            'pulse_width': self.pulse_width,
            'galactic_dispersion': self.galactic_dispersion,
            'host_dispersion': self.host_dispersion,
            'igm_dispersion': self.igm_dispersion,
            'source_dispersion': self.source_dispersion,
            'spectral_index': self.spectral_index,
            'time': self.time,
        }

        if 'observations' in self.__dict__.keys():

            out_dict['observations'] = numpy.array([*self.observations.keys()])

            for obs in out_dict['observations']:

                out_dict.update(
                    self.observations[obs].to_dict(key=obs)
                )

        numpy.savez(file, **out_dict)

    @staticmethod
    def load(file):

        output = FastRadioBursts(n_frb=0, verbose=False)

        input_file = numpy.load(file, allow_pickle=True)
        udm = units.pc / units.cm**3
        ergs = units.erg / units.s

        output.L_0 = input_file['L_0'] * ergs
        output.Lstar = input_file['Lstar'] * ergs
        output.phistar = input_file['phistar'] * output.phistar.unit
        output.ra_range = input_file['ra_range'] * output.ra_range.unit
        output.dec_range = input_file['dec_range'] * output.dec_range.unit
        output.lower_frequency = input_file['lower_frequency'] * units.MHz
        output.higher_frequency = input_file['higher_frequency'] * units.MHz
        output.sky_rate = input_file['sky_rate'] / units.day
        output.area = input_file['area'] * units.degree**2
        output.rate = input_file['rate'] / units.day
        output.duration = input_file['duration'] * output.duration.unit

        ra = input_file['ra'] * units.degree
        dec = input_file['dec'] * units.degree

        output.sky_coord = coordinates.SkyCoord(ra, dec, frame='icrs')

        output.redshift = input_file['redshift']
        output.comoving_distance = input_file['comoving_distance'] * units.Mpc
        output.luminosity = input_file['luminosity'] * ergs
        output.pulse_width = input_file['pulse_width'] * units.ms
        output.time = input_file['time'] * units.ms
        output.spectral_index = input_file['spectral_index']

        output.n_frb = len(output.redshift)

        _zp1 = 1 + output.redshift
        _sip1 = 1 + output.spectral_index

        output.arrived_pulse_width = output.pulse_width * _zp1
        output.luminosity_distance = output.comoving_distance * _zp1

        surface = 4 * numpy.pi * output.luminosity_distance**2

        output.flux = (output.luminosity / surface).to(units.Jy * units.MHz)

        nu_low = input_file['lower_frequency']**_sip1
        nu_high = input_file['higher_frequency']**_sip1

        output.S0 = _sip1 * output.flux / (nu_high - nu_low)
        output.S0 = output.S0 / units.MHz

        gal_DM = input_file['galactic_dispersion'] * udm
        host_DM = input_file['host_dispersion'] * udm
        src_DM = input_file['source_dispersion'] * udm
        igm_DM = input_file['igm_dispersion'] * udm

        egal_DM = igm_DM + (src_DM + host_DM) / _zp1

        output.galactic_dispersion = gal_DM
        output.host_dispersion = host_DM
        output.source_dispersion = src_DM
        output.igm_dispersion = igm_DM

        output.extra_galactic_dispersion = egal_DM
        output.dispersion = gal_DM + egal_DM

        if 'observations' in input_file.files:

            output.observations = {}

            for obs in input_file['observations']:

                output.observations[obs] = Observation.from_dict(key=obs,
                                                                 **input_file)

        return output
