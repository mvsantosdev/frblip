import sys

import numpy
import pandas

from numpy import random

from scipy.special import comb, hyp2f1
from scipy.integrate import quad, cumtrapz

from operator import itemgetter
from itertools import combinations, product
from functools import cached_property, partial

from astropy.time import Time
from astropy import units, constants, coordinates
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator

from .utils import _DATA, load_params, load_file
from .utils import paired_shapes, sub_dict, xfactors

from .distributions import Redshift, Schechter

from .dispersion_measure import Galactic, IGM

from .observation import Observation, Interferometry
from .cosmology import Cosmology, builtin


class FastRadioBursts(object):

    """
    Class which defines a Fast Radio Burst population
    """

    def __init__(self, n_frb=None, days=1, log_Lstar=44.46, log_L0=41.96,
                 phistar=339, alpha=-1.79, wint=(.13, .33), si=(-15, 15),
                 ra=(0, 24), dec=(-90, 90), zmin=0, zmax=6, start=None,
                 lower_frequency=400, higher_frequency=1400,
                 gal_method='yt2020_analytic', gal_nside=128,
                 cosmology='Planck_18', verbose=True):

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
        sys.stdout = old_target if verbose else open(os.devnull, 'w')

        self.__load_params(n_frb, days, log_Lstar, log_L0, phistar,
                           alpha, wint, si, ra, dec, zmin, zmax, cosmology,
                           lower_frequency, higher_frequency, start,
                           gal_nside, gal_method)
        self.__frb_rate(n_frb, days)
        self.__random()

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

    def select(self, idx, inplace=False):

        select_dict = {
            name: attr[idx]
            for name, attr in self.__dict__.items()
            if hasattr(attr, 'size')
            and attr.size == self.n_frb
        }

        select_dict['n_frb'] = select_dict['redshift'].size

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

    @cached_property
    def __sky_rate(self):
        Lum = self.phistar / self.__lumdist.pdf_norm
        Vol = 1 / self.__zdist.pdf_norm
        return (Lum * Vol).to(1 / units.day)

    @cached_property
    def redshift(self):
        return self.__zdist.rvs(size=self.n_frb)

    @cached_property
    def log_luminosity(self):
        loglum = self.__lumdist.log_rvs(size=self.n_frb)
        return loglum * units.LogUnit() + self.log_Lstar

    @cached_property
    def pulse_width(self):
        width = random.lognormal(self.w_mean, self.w_std, size=self.n_frb)
        return width * units.ms

    @cached_property
    def itrs_time(self):
        time_ms = int(self.duration.to(units.ms).value)
        dt = random.randint(time_ms, size=self.n_frb) * units.ms
        return self.start + dt

    @cached_property
    def spectral_index(self):
        return random.uniform(self.si_min, self.si_max, self.n_frb)

    @cached_property
    def icrs(self):
        sin = numpy.sin(self.dec_range)
        args = random.uniform(*sin, self.n_frb)
        decs = numpy.arcsin(args) * units.rad
        decs = decs.to(units.degree)
        ras = random.uniform(*self.ra_range.value, self.n_frb)
        ras = ras * self.ra_range.unit
        return coordinates.SkyCoord(ras, decs, frame='icrs')

    @cached_property
    def area(self):
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
    def __arrived_pulse_width(self):
        return (1 + self.redshift) * self.pulse_width

    @cached_property
    def __S0(self):
        _sip1 = self.spectral_index + 1
        nu_lp = (self.lower_frequency / units.MHz)**_sip1
        nu_hp = (self.higher_frequency / units.MHz)**_sip1
        return self.__flux / (nu_hp - nu_lp)

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
        return Galactic(self.gal_nside, self.gal_method)

    @cached_property
    def __igm_dm(self):
        return IGM(self.__cosmology)

    @cached_property
    def galactic_dm(self):
        gl = self.galactic.l
        gb = self.galactic.b
        return self.__gal_dm(gl, gb)

    @cached_property
    def igm_dm(self):
        z = self.redshift
        return self.__igm_dm(z)

    def obstime(self, location):

        loc = location.get_itrs()
        loc = loc.cartesian.xyz

        path = loc @ self.xyz
        time_delay = path / constants.c

        return self.itrs_time - time_delay

    def altaz(self, location, interp=300):

        obstime = self.obstime(location)
        frame = coordinates.AltAz(location=location, obstime=obstime)
        interp_time = interp * units.s

        with erfa_astrom.set(ErfaAstromInterpolator(interp_time)):
            return self.icrs.transform_to(frame)

    def __frb_rate(self, n_frb, days):

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
            sky_fraction = self.area / units.astrophys.sp
            self.rate = self.__sky_rate * sky_fraction.to(1)
        else:
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

    def __load_params(self, n_frb, days, log_Lstar, log_L0, phistar,
                      alpha, wint, si, ra, dec, zmin, zmax, cosmology,
                      lower_frequency, higher_frequency, start,
                      gal_nside, gal_method):

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
        self.cosmology = cosmology

        self.start = Time.now() if start is None else start

        self.gal_nside = gal_nside
        self.gal_method = gal_method

    def __random(self):

        self.icrs
        self.area
        self.redshift
        self.itrs_time
        self.pulse_width
        self.log_luminosity
        self.spectral_index

    def __observe(self, telescope, location=None, start=None,
                  name=None, altaz=None):

        if 'observations' not in self.__dict__:
            self.observations = {}

        noise = telescope.noise()

        location = telescope.location if location is None else location

        frequency_bands = telescope.frequency_bands
        sampling_time = telescope.sampling_time

        altaz = self.altaz(location) if altaz is None else altaz
        response = telescope.response(altaz)
        array = telescope.array

        if array is not None:
            xyz = altaz.cartesian.xyz[:2]
            time_array = array @ xyz / constants.c
            time_array = time_array.to(units.ms)
        else:
            time_array = None

        observation = Observation(response, noise, frequency_bands,
                                  sampling_time, altaz, time_array)

        n_obs = len(self.observations)
        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        self.observations[obs_name] = observation

    def observe(self, telescopes, location=None, start=None,
                name=None, altaz=None):

        if type(telescopes) is dict:
            for name, telescope in telescopes.items():
                self.__observe(telescope, location, start, name, altaz)
        else:
            self.__observe(telescopes, location, start, name, altaz)

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

    def __response(self, name, channels=False):

        observation = self[name]
        response = observation.get_response(self.spectral_index, channels)
        return response

    def __signal(self, name, channels=False):

        response = self.__response(name, channels)
        signal = response.T * self.__S0
        return signal.T

    def __noise(self, name, channels=False):

        observation = self[name]
        return observation.get_noise(channels)

    def __sensitivity(self, name, channels=False):

        return 1 / self.__response_to_noise(name, channels)

    def __response_to_noise(self, name, channels=False):

        response = self.__response(name, channels)
        noise = self.__noise(name, channels)

        return response / noise

    def __signal_to_noise(self, name, channels=False):

        signal = self.__signal(name, channels)
        noise = self.__noise(name, channels)

        return (signal / noise).value

    def __counts(self, name, channels=False, SNR=None, total=False):

        S = numpy.arange(1, 11).reshape(-1, 1) if SNR is None else SNR
        snr = self.__signal_to_noise(name, channels)
        snr = numpy.nan_to_num(snr, nan=0.0, posinf=0.0)

        if total:
            axes = range(1, snr.ndim)
            snr = numpy.apply_over_axes(numpy.max, snr, axes).ravel()

        if snr.ndim == 1:
            return (snr > S).sum(-1)

        shape = snr.shape[1:]
        n_beams = numpy.prod(shape)
        snr = snr.reshape((-1, n_beams))

        counts = numpy.stack([
            (col > S).sum(-1)
            for col in snr.T
        ])

        return counts.reshape(*shape, -1)

    def __get(self, func_name=None, names=None,
              channels=False, **kwargs):

        func = self.__getattribute__(func_name)

        if names is None:
            observations = self.observations
        elif isinstance(names, str):
            return func(names, channels, **kwargs)
        else:
            values = itemgetter(*names)(self.observations)
            observations = dict(zip(names, values))

        return {
            obs: func(obs, channels, **kwargs)
            for obs in observations
        }

    def response(self, names=None, channels=False):

        return self.__get('_FastRadioBursts__response', names, channels)

    def signal(self, names=None, channels=False):

        return self.__get('_FastRadioBursts__signal', names, channels)

    def noise(self, names=None, channels=False):

        return self.__get('_FastRadioBursts__noise', names, channels)

    def response_to_noise(self, names=None, channels=False):

        return self.__get('_FastRadioBursts__response_to_noise',
                          names, channels)

    def sensitivity(self, names=None, channels=False):

        return self.__get('_FastRadioBursts__sensitivity',
                          names, channels)

    def signal_to_noise(self, names=None, channels=False):

        return self.__get('_FastRadioBursts__signal_to_noise',
                          names, channels)

    def counts(self, names=None, channels=False, total=False):

        return self.__get('_FastRadioBursts__counts', names,
                          channels, total=total)

    def interferometry(self, *names, time_delay=True):

        key = '_'.join(names)
        key = 'INTF_{}'.format(key)
        observations = [self[name] for name in names]
        interferometry = Interferometry(*observations, time_delay=time_delay)
        self.observations[key] = interferometry

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
            if "_FastRadioBursts__" not in key
        }

        icrs = out_dict.pop('icrs', None)
        if icrs:
            out_dict['ra'] = icrs.ra
            out_dict['dec'] = icrs.dec

        observations = out_dict.pop('observations', None)
        if observations is not None:
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
            and value.unit != ''
        })

        out_dict.update({
            key: value.value
            for key, value in out_dict.items()
            if hasattr(value, 'unit')
        })

        keys = [*out_dict.keys()]

        for name in keys:
            attr = out_dict[name]
            if type(attr) is COO:
                coo = out_dict.pop(name)
                out_dict['{}_coords'.format(name)] = coo.coords
                out_dict['{}_data'.format(name)] = coo.data
                out_dict['{}_shape'.format(name)] = coo.shape

        return out_dict

    def save(self, file, compressed=True):
        out_dict = self.to_dict()
        if compressed:
            numpy.savez_compressed(file, **out_dict)
        else:
            numpy.savez(file, **out_dict)

    @staticmethod
    def from_dict(params):

        output = FastRadioBursts.__new__(FastRadioBursts)
        observations = params.pop('observations', None)

        if observations is not None:

            output.observations = {}

            for name in observations:
                flag = '{}__'.format(name)
                obs = sub_dict(params, flag=flag, pop=True)
                obs['response'] = obs.pop('response')
                observation = Observation.from_dict(obs)
                output.observations[name] = observation

        ra = params.pop('ra', None)
        dec = params.pop('dec', None)
        if ra and dec:
            icrs = coordinates.SkyCoord(ra, dec, frame='icrs')
            params['icrs'] = icrs

        output.__dict__.update(params)

        return output

    @staticmethod
    def load(file):
        input_dict = load_file(file)
        params = load_params(input_dict)
        return FastRadioBursts.from_dict(params)
