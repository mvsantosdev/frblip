import os
import sys

import bz2
import dill

import numpy
import xarray

from numpy import random

from operator import itemgetter
from functools import cached_property

from astropy.time import Time
from astropy import units, constants, coordinates
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator

from .random import Redshift, Schechter, SpectralIndex

from .random.dispersion_measure import GalacticDM
from .random.dispersion_measure import InterGalacticDM, HostGalaxyDM

from .observation import Observation, Interferometry
from .cosmology import Cosmology, builtin


class FastRadioBursts(object):

    """Class which defines a Fast Radio Burst population"""

    def __init__(self, n_frb=None, days=1, log_Lstar=44.46, log_L0=41.96,
                 phistar=339, gamma=-1.79, pulse_width=(-6.917, 0.824),
                 zmin=0, zmax=6, ra=(0, 24), dec=(-90, 90), start=None,
                 low_frequency=10.0, high_frequency=10000.0,
                 low_frequency_cal=400.0, high_frequency_cal=1400.0,
                 emission_frame=True, spectral_index='CHIME2021',
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

        self.__load_params(n_frb, days, log_Lstar, log_L0, phistar, gamma,
                           pulse_width, zmin, zmax, ra, dec, start,
                           low_frequency, high_frequency,
                           low_frequency_cal, high_frequency_cal,
                           emission_frame, spectral_index, gal_method,
                           gal_nside, host_source, host_model,
                           cosmology, free_electron_bias)
        self.__frb_rate(n_frb, days)
        self.__S0

        sys.stdout = old_target

    def __load_params(self, n_frb, days, log_Lstar, log_L0, phistar, gamma,
                      pulse_width, zmin, zmax, ra, dec, start, low_frequency,
                      high_frequency, low_frequency_cal, high_frequency_cal,
                      emission_frame, spectral_index, gal_method, gal_nside,
                      host_source, host_model, cosmology, free_electron_bias):

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
        return (width * units.s).to(units.ms)

    @cached_property
    def emitted_pulse_width(self):
        """ """
        return self.pulse_width / (1 + self.redshift)

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
        return self.__spec_idx_dist.rvs(self.n_frb)

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

    def shuffle(self):

        idx = numpy.arange(self.n_frb)
        numpy.random.shuffle(idx)

        self.__dict__.update({
            key: value[idx]
            for key, value in self.__dict__.items()
            if key not in ('icrs', 'itrs', 'itrs_time')
            and numpy.size(value) == self.n_frb
        })

    def __observe(self, telescope, location=None, name=None, altaz=None):

        print('Performing observation for telescope {}...'.format(name))

        if 'observations' not in self.__dict__:
            self.observations = {}

        n_obs = len(self.observations)
        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        location = telescope.location if location is None else location
        lon, lat, height = location.lon, location.lat, location.height

        print(
            'Computing positions for {} FRB'.format(self.n_frb),
            'at site lon={:.3f}, lat={:.3f},'.format(lon, lat),
            'height={:.3f}.'.format(height), end='\n\n'
        )

        sampling_time = telescope.sampling_time
        frequency_range = telescope.frequency_range
        width = telescope.frequency_range.diff()

        zmax = self.high_frequency / frequency_range[0] - 1
        zmin = self.low_frequency / frequency_range[-1] - 1
        zmin = zmin.clip(0)

        altaz = self.altaz(location) if altaz is None else altaz
        in_range = (self.redshift > zmin) & (self.redshift < zmax)
        visible = altaz.alt > 0
        mask = visible & in_range

        vis_frac = 100 * visible.mean()
        range_frac = 100 * in_range.mean()
        obs_frac = 100 * mask.mean()

        print('>>> {:.3}% are visible.'.format(vis_frac))
        print('>>> {:.3}% are in frequency range.'.format(range_frac))
        print('>>> {:.3}% are observable.'.format(obs_frac), end='\n\n')

        resp = telescope.response(altaz[mask])
        response = numpy.zeros((self.n_frb, *resp.shape[1:]))
        response[mask] = resp

        S0 = self.__S0 / width
        S0 = (S0 / units.Jy).to(1)
        S0 = xarray.DataArray(S0, dims='FRB')
        response = xarray.DataArray(response, dims=('FRB', obs_name))
        response = response * S0
        response.name = 'Response'

        noise = telescope.noise
        noise = (noise / units.Jy).to(1)
        noise = xarray.DataArray(noise, dims=obs_name, name='Noise')
        noise.name = 'Noise'

        time_array = telescope.time_array(altaz)
        time_array = (time_array * units.MHz).to(1)
        time_array = xarray.DataArray(time_array, dims=('FRB', obs_name))
        time_array.name = 'Time Array'

        observation = Observation(response, noise, altaz, time_array,
                                  frequency_range, sampling_time)

        self.observations[obs_name] = observation

    def observe(self, telescopes, location=None, name=None,
                altaz=None, verbose=True):
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
                self.__observe(telescope, location, name, altaz)
        else:
            self.__observe(telescopes, location, name, altaz)

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

    def counts(self, names=None, channels=1, SNR=None,
               total=False, level=None):
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

        return self.__get('_FastRadioBursts__counts', names, channels,
                          SNR=SNR, total=total, level=level)

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

    def interferometry(self, namei, namej=None, degradation=None):
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
        interferometry = Interferometry(obsi, obsj, degradation)
        self.observations[key] = interferometry

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
        dill.dump(self, file, dill.HIGHEST_PROTOCOL)
        file.close()

    @staticmethod
    def load(name):
        """
        Parameters
        ----------
        name :


        Returns
        -------
        """
        file_name = '{}.blips'.format(name)
        file = bz2.BZ2File(file_name, 'rb')
        loaded = dill.load(file)
        file.close()
        return loaded
