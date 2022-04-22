import os
import sys

import numpy
import xarray

from numpy import random

from operator import itemgetter
from functools import cached_property

from astropy.time import Time
from astropy import units, constants, coordinates
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator

from .utils import load_params, load_file, sub_dict

from .random import Redshift, Schechter

from .random.dispersion_measure import GalacticDM
from .random.dispersion_measure import InterGalacticDM, HostGalaxyDM

from .observation import Observation, Interferometry
from .cosmology import Cosmology, builtin


class FastRadioBursts(object):

    """Class which defines a Fast Radio Burst population"""

    def __init__(self, n_frb=None, days=1, log_Lstar=44.46, log_L0=41.96,
                 phistar=339, gamma=-1.79, wint=(.13, .33), si=(-5, 5),
                 zmin=0, zmax=6, ra=(0, 24), dec=(-90, 90), start=None,
                 low_frequency=10.0, high_frequency=10000.0,
                 low_frequency_cal=400.0, high_frequency_cal=1400.0,
                 gal_method='yt2020_analytic', gal_nside=128,
                 host_source='luo18', host_model=('ALG', 'YMW16'),
                 cosmology='Planck_18', free_electron_bias='Takahashi2021',
                 verbose=True):

        """
        Creates a FRB population object.
        Parameters
        ----------
        n_frb : int
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

        self.__load_params(n_frb, days, log_Lstar, log_L0, phistar,
                           gamma, wint, si, ra, dec, zmin, zmax, cosmology,
                           low_frequency, high_frequency, start,
                           low_frequency_cal, high_frequency_cal,
                           gal_nside, gal_method, host_source, host_model,
                           free_electron_bias)
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

        stop = self.n_frb if stop is None else stop
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
        return self.__zdist.rvs(size=self.n_frb)

    @cached_property
    def log_luminosity(self):
        """ """
        loglum = self.__lumdist.log_rvs(size=self.n_frb)
        return loglum * units.LogUnit() + self.log_Lstar

    @cached_property
    def pulse_width(self):
        """ """
        width = random.lognormal(self.w_mean, self.w_std, size=self.n_frb)
        return width * units.ms

    @cached_property
    def emitted_pulse_width(self):
        """ """
        return (1 + self.redshift) * self.pulse_width

    @cached_property
    def itrs_time(self):
        """ """
        time_ms = int(self.duration.to(units.us).value)
        dt = random.randint(time_ms, size=self.n_frb)
        dt = numpy.sort(dt) * units.us
        return self.start + dt

    @cached_property
    def spectral_index(self):
        """ """
        if hasattr(self, 'si_min') and hasattr(self, 'si_max'):
            return random.uniform(self.si_min, self.si_max, self.n_frb)
        return numpy.full(self.n_frb, self.si)

    @cached_property
    def icrs(self):
        """ """
        sin = numpy.sin(self.dec_range)
        args = random.uniform(*sin, self.n_frb)
        decs = numpy.arcsin(args) * units.rad
        decs = decs.to(units.degree)
        ras = random.uniform(*self.ra_range.value, self.n_frb)
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
        z_factor = (1 + self.redshift)**_sip1
        nu_lp = (self.low_frequency_cal / units.MHz)**_sip1
        nu_hp = (self.high_frequency_cal / units.MHz)**_sip1
        return self.__flux * z_factor / (nu_hp - nu_lp)

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
                            self.__cosmology)

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
            sky_fraction = self.area / units.spat
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
                      gamma, wint, si, ra, dec, zmin, zmax, cosmology,
                      low_frequency, high_frequency, start,
                      low_frequency_cal, high_frequency_cal,
                      gal_nside, gal_method, host_source, host_model,
                      free_electron_bias):

        self.zmin = zmin
        self.zmax = zmax
        self.log_L0 = log_L0 * units.LogUnit(units.erg / units.s)
        self.log_Lstar = log_Lstar * units.LogUnit(units.erg / units.s)
        self.phistar = phistar / (units.Gpc**3 * units.year)
        self.gamma = gamma
        self.w_mean, self.w_std = wint
        if isinstance(si, (int, float)):
            self.si = si
        else:
            self.si_min, self.si_max = si
        self.ra_range = numpy.array(ra) * units.hourangle
        self.dec_range = numpy.array(dec) * units.degree
        self.low_frequency = low_frequency * units.MHz
        self.high_frequency = high_frequency * units.MHz
        self.low_frequency_cal = low_frequency_cal * units.MHz
        self.high_frequency_cal = high_frequency_cal * units.MHz
        self.free_electron_bias = free_electron_bias
        self.cosmology = cosmology
        self.host_source = host_source
        self.host_model = host_model

        if start is None:
            now = Time.now()
            today = now.iso.split()
            self.start = Time(today[0])
        else:
            self.start = start

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

        print('Performing observation for telescope {}...'.format(name))

        if 'observations' not in self.__dict__:
            self.observations = {}

        location = telescope.location if location is None else location
        lon, lat, height = location.lon, location.lat, location.height

        print(
            'Computing positions for {} FRB'.format(self.n_frb),
            'at site lon={:.3f}, lat={:.3f},'.format(lon, lat),
            'height={:.3f}.'.format(height), end='\n\n'
        )

        channels = telescope.channels
        frequency_range = telescope.frequency_range
        sampling_time = telescope.sampling_time

        zmax = self.high_frequency / frequency_range[0] - 1
        zmin = self.low_frequency / frequency_range[-1] - 1
        zmin = zmin.clip(0)

        altaz = self.altaz(location) if altaz is None else altaz
        in_range = (self.redshift > zmin) & (self.redshift < zmax)
        visible = altaz.alt > 0
        obs = visible & in_range

        vis_frac = 100 * visible.mean()
        range_frac = 100 * in_range.mean()
        obs_frac = 100 * obs.mean()

        print('>>> {:.3}% are visible.'.format(vis_frac))
        print('>>> {:.3}% are in frequency range.'.format(range_frac))
        print('>>> {:.3}% are observable.'.format(obs_frac), end='\n\n')

        resp = telescope.response(altaz[obs])
        response = numpy.zeros((self.n_frb, *resp.shape[1:]))
        response[obs] = resp

        array = telescope.array

        if array is not None:
            xyz = altaz.cartesian.xyz[:2]
            time_array = array @ xyz / constants.c
            time_array = time_array.T.to(units.ms)
        else:
            time_array = None

        n_obs = len(self.observations)
        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        width = telescope.frequency_range.diff()

        S0 = self.__S0 / width
        S0 = (S0 / units.Jy).to(1)
        S0 = xarray.DataArray(S0, dims='FRB')
        response = xarray.DataArray(response, dims=('FRB', obs_name))
        response = response * S0
        response.name = 'Response'

        noise = telescope.noise()
        noise = (noise / units.Jy).to(1)
        noise = xarray.DataArray(noise, dims=obs_name, name='Noise')

        observation = Observation(response, noise, channels,
                                  frequency_range, sampling_time,
                                  altaz, time_array)

        self.observations[obs_name] = observation

    def observe(self, telescopes, location=None, start=None,
                name=None, altaz=None, verbose=True):
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
                self.__observe(telescope, location, start, name, altaz)
        else:
            self.__observe(telescopes, location, start, name, altaz)

        sys.stdout = old_target

    def clear(self, names=None):
        """

        Parameters
        ----------
        names :
            (Default value = None)

        Returns
        -------


        """

        if names is None:
            del self.observations
        elif isinstance(names, str):
            del self.observations[names]
        else:
            for name in names:
                del self.observations[name]

    def __signal(self, name, channels=1):

        observation = self[name]
        spectral_index = self.spectral_index
        return observation.get_response(spectral_index, channels)

    def __noise(self, name, channels=1):

        observation = self[name]
        return observation.get_noise(channels)

    def __signal_to_noise(self, name, channels=1, total=False, level=None):

        signal = self.__signal(name, channels)
        noise = self.__noise(name, channels)

        snr = signal / noise

        if total:
            lvl = snr.dims[1:-1]
            return snr.max(lvl)
        if level is not None:
            return snr.max(level)
        return snr

    def __detected(self, name, channels=1, SNR=None, total=False,
                   level=None):

        snr = self.__signal_to_noise(name, channels, total, level)
        S = numpy.arange(1, 11) if SNR is None else SNR
        S = xarray.DataArray(S, dims='SNR')
        return snr > S

    def __counts(self, name, channels=1, SNR=None, total=False,
                 level=None):

        detected = self.__detected(name, channels, SNR, total, level)
        return detected.sum('FRB')

    def __baselines_counts(self, name, channels=1, reference='MAIN'):

        ref_detections = self.detected(reference, total=True)
        beams = self[name].response.shape[-1]
        number = beams * (beams - 1) // 2
        base_lines = numpy.arange(number + 1)
        base_lines = xarray.DataArray(base_lines, dims='BASELINES')

        key = 'INTF_{}_{}'.format(reference, name)
        if key not in self.observations:
            self.interferometry(reference, name)

        detections = self.detected(key, level=reference)
        detections = (ref_detections * detections).sum(name)
        count = (detections > base_lines).sum('FRB')

        if beams > 1:
            key = 'INTF_{}'.format(name)
            if key not in self.observations:
                self.interferometry(name)
            detections = self.detected(key)
            detections = (ref_detections * detections).sum(name)
            count += (detections > base_lines).sum('FRB')

        return count

    def __get(self, func_name=None, names=None, channels=1, **kwargs):

        func = self.__getattribute__(func_name)

        if names is None:
            names = self.observations.keys()
        elif isinstance(names, str):
            return func(names, channels, **kwargs)

        return {
            name: func(name, channels, **kwargs)
            for name in names
        }

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
                        level=None):
        """

        Parameters
        ----------
        names :
            (Default value = None)
        channels :
            (Default value = False)
        total :
            (Default value = False)
        level :
            (Default value = None)

        Returns
        -------


        """

        return self.__get('_FastRadioBursts__signal_to_noise', names,
                          channels, total=total, level=level)

    def detected(self, names=None, channels=1, total=False, level=None):
        """

        Parameters
        ----------
        names :
            (Default value = None)
        channels :
            (Default value = False)
        total :
            (Default value = False)
        level :
            (Default value = None)

        Returns
        -------


        """

        return self.__get('_FastRadioBursts__detected', names,
                          channels, total=total, level=level)

    def counts(self, names=None, channels=1, total=False, level=None):
        """

        Parameters
        ----------
        names :
            (Default value = None)
        channels :
            (Default value = False)
        total :
            (Default value = False)
        level :
            (Default value = None)

        Returns
        -------


        """

        return self.__get('_FastRadioBursts__counts', names,
                          channels, total=total, level=level)

    def baselines_counts(self, names=None, channels=1, reference='MAIN'):

        """

        Parameters
        ----------
        names :
            (Default value = None)
        channels :
            (Default value = False)
        reference :
            (Default value = 'MAIN')

        Returns
        -------


        """

        return self.__get('_FastRadioBursts__baselines_counts',
                          names, channels, reference=reference)

    def interferometry(self, namei, namej=None, time_delay=False):
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

        obsi, obsj = self[namei], self[namej]
        if namej is None:
            key = 'INTF_{}'.format(namei)
        else:
            key = 'INTF_{}_{}'.format(namei, namej)
        interferometry = Interferometry(obsi, obsj, time_delay=time_delay)
        self.observations[key] = interferometry

    def to_dict(self):
        """ """

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

        out_dict['unit'] = {
            key: value.unit.to_string()
            for key, value in out_dict.items()
            if hasattr(value, 'unit')
            and value.unit != ''
        }

        out_dict.update({
            'u_{}'.format(key): value.unit.to_string()
            for key, value in out_dict.items()
            if hasattr(value, 'unit')
            and value.unit != ''
        })

        return out_dict

    def save(self, file, compressed=True):
        """

        Parameters
        ----------
        file :

        compressed :
            (Default value = True)

        Returns
        -------


        """
        out_dict = self.to_dict()
        if compressed:
            numpy.savez_compressed(file, **out_dict)
        else:
            numpy.savez(file, **out_dict)

    @staticmethod
    def from_dict(params):
        """

        Parameters
        ----------
        params :


        Returns
        -------


        """

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
        """

        Parameters
        ----------
        file :


        Returns
        -------


        """
        input_dict = load_file(file)
        params = load_params(input_dict)
        return FastRadioBursts.from_dict(params)
