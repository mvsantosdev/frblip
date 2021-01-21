import sys

import numpy

from numpy import random

import pandas

from scipy.special import comb

from itertools import combinations

from scipy.integrate import quad, cumtrapz

from astropy.time import Time

from astropy import cosmology, coordinates, units

from .utils import _all_sky_area, _DATA, schechter, rvs_from_cdf, simps

from .dispersion import *

from .observation import Observation


class FastRadioBursts():

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

    def get_local_coordinates(self, location, start_time=None):

        if start_time is None:
            start_time = Time.now()

        date_time = start_time + self.time.ravel()

        AltAzCoords = coordinates.AltAz(location=location,
                                        obstime=date_time)

        return self.sky_coord.transform_to(AltAzCoords)

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

    def observe(self, Telescope, location=None, start_time=None,
                name=None, local_coordinates=None):

        if 'observations' not in self.__dict__.keys():

            self.observations = {}

        n_obs = len(self.observations)

        location = Telescope.location if location is None else location
        start_time = Telescope.start_time if start_time is None else start_time

        if local_coordinates is None:

            local_coordinates = self.get_local_coordinates(location,
                                                           start_time)

        response = Telescope.selection(local_coordinates.az,
                                       local_coordinates.alt)

        noise = Telescope.minimum_temperature / Telescope.gain.reshape(-1, 1)
        noise = noise.to(units.Jy)

        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        obs = Observation(response, noise, Telescope.frequency_bands,
                          Telescope.sampling_time, local_coordinates)

        self.observations[obs_name] = obs

    def _signal(self, name, channels=False):

        obs = self.observations[name]

        signal = self.specific_flux(obs._frequency)
        signal = simps(signal)

        nshape = len(obs.response.shape) - 1
        ishape = numpy.ones(nshape, dtype=int)

        signal = obs.response[..., numpy.newaxis] * signal[:, numpy.newaxis]

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

    def _cross_correlation(self, namei, namej):

        """
        #Compute the cross correlations between two observation sets
        """

        obsi = self.observations[namei]
        obsj = self.observations[namej]

        respi = obsi.response[..., numpy.newaxis]
        respj = obsj.response[:, numpy.newaxis]

        noii = obsi.noise[:, numpy.newaxis]
        noij = obsj.noise

        response = numpy.sqrt(respi * respj)
        noise = numpy.sqrt(0.5 * noii * noij)

        return Observation(response, noise, obsi.frequency_bands)

    def interferometry(self, *names):

        """
        #Perform a interferometry observation by a Radio Telescope array.
        """

        n_obs = len(names)

        if n_obs == 2:

            out = self._cross_correlation(*names, interference)
            out.response = numpy.squeeze(out.response)
            out.noise = numpy.squeeze(out.noise)

            self.observations['INTF_{}_{}'.format(*names)] = out

        elif n_obs > 2:

            n_channel = numpy.unique([
                self.observations[name].n_channel
                for name in names
            ]).astype(numpy.int).item(0)

            n_beam = numpy.concatenate([
                self.observations[name].n_beam
                for name in names
            ]).astype(numpy.int)

            obsij = [
                self._cross_correlation(namei, namej)
                for namei, namej in combinations(names, 2)
            ]

            n_comb = comb(n_obs, 2, exact=True)

            shapes = numpy.ones((n_comb, n_obs), dtype=numpy.int)

            idx_beam = numpy.row_stack([*combinations(range(n_obs), 2)])
            idx_comb = numpy.tile(numpy.arange(n_comb).reshape(-1, 1), (1, 2))

            shapes[idx_comb, idx_beam] = n_beam[idx_beam]

            response = [
                obs.response.reshape((self.n_frb, *shape))
                for shape, obs in zip(shapes, obsij)
            ]

            response = sum(response, numpy.empty(()))

            noise = [
                1 / obs.noise.value.reshape((*shape, n_channel))**2
                for shape, obs in zip(shapes, obsij)
            ]

            noise = sum(noise, numpy.empty(()))
            noise = numpy.sqrt(1 / noise)

            frequency_bands = self.observations[names[0]].frequency_bands

            obs = Observation(response, noise, frequency_bands)

            obs.response = numpy.squeeze(obs.response)
            obs.noise = numpy.squeeze(obs.noise)

            labels, counts = np.unique(names, return_counts=True)

            key = ['{}x{}'.format(c, l) for c, l in zip(counts, labels)]
            key = '_'.join(key).replace('1x', '')
            key = 'INTF_{}'.format(key)

            self.observations[key] = obs

    def split_beams(self, name):

        observation = self.observations.pop(name)
        splitted = observation.split_beams()

        self.observations.update({
            '{}_BEAM_{}'.format(name, beam): obs
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

                lon = self.observations[obs].coordinates.location.lon
                lat = self.observations[obs].coordinates.location.lat
                height = self.observations[obs].coordinates.location.height

                az = self.observations[obs].coordinates.az
                alt = self.observations[obs].coordinates.alt

                obstime = self.observations[obs].coordinates.obstime.iso

                response = self.observations[obs].response
                noise = self.observations[obs].noise
                sampling_time = self.observations[obs].sampling_time
                frequency_bands = self.observations[obs].frequency_bands

                out_dict.update({
                    '{}__az'.format(obs): az,
                    '{}__alt'.format(obs): alt,
                    '{}__obstime'.format(obs): obstime,
                    '{}__lon'.format(obs): lon,
                    '{}__lat'.format(obs): lat,
                    '{}__height'.format(obs): height,
                    '{}__response'.format(obs): response,
                    '{}__noise'.format(obs): noise,
                    '{}__sampling_time'.format(obs): sampling_time,
                    '{}__frequency_bands'.format(obs): frequency_bands
                })

        numpy.savez(file, **out_dict)

    @staticmethod
    def load(file):

        output = FastRadioBursts(n_frb=0, verbose=False)

        input_file = numpy.load(file)
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

                lon = input_file['{}__lon'.format(obs)] * units.degree
                lat = input_file['{}__lat'.format(obs)] * units.degree
                height = input_file['{}__height'.format(obs)] * units.meter

                location = coordinates.EarthLocation(lon=lon, lat=lat,
                                                     height=height)

                az = input_file['{}__az'.format(obs)] * units.degree
                alt = input_file['{}__alt'.format(obs)] * units.degree
                obstime = Time(input_file['{}__obstime'.format(obs)])

                local_coordinates = coordinates.AltAz(az=az, alt=alt,
                                                      location=location,
                                                      obstime=obstime)

                frequency_bands = input_file['{}__frequency_bands'.format(obs)]
                frequency_bands = frequency_bands * units.MHz

                sampling_time = input_file['{}__sampling_time'.format(obs)]
                sampling_time = sampling_time * units.ms

                response = input_file['{}__response'.format(obs)]
                noise = input_file['{}__noise'.format(obs)] * units.Jy

                output.observations[obs] = Observation(response, noise,
                                                       frequency_bands,
                                                       sampling_time,
                                                       local_coordinates)

        return output
