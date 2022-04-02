import warnings

import numpy
from astropy_healpix import HEALPix

from astropy.time import Time
from astropy import coordinates, units, constants
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator

from functools import cached_property

from .random import Redshift, Schechter
from .observation import Observation, Interferometry
from .cosmology import Cosmology, builtin


class HealPixMap(HEALPix):
    """ """

    def __init__(self, nside=None, order='ring', phistar=339,
                 alpha=-1.79, log_Lstar=44.46, log_L0=41.96,
                 lower_frequency=400, higher_frequency=1400,
                 cosmology='Planck_18', zmin=0.0, zmax=6.0):

        super().__init__(nside, order)

        self.__load_params(phistar, alpha, log_Lstar, log_L0,
                           lower_frequency, higher_frequency,
                           cosmology, zmin, zmax)

    def __load_params(self, phistar, alpha, log_Lstar, log_L0,
                      lower_frequency, higher_frequency,
                      cosmology, zmin, zmax):

        self.log_L0 = log_L0 * units.LogUnit(units.erg / units.s)
        self.log_Lstar = log_Lstar * units.LogUnit(units.erg / units.s)
        self.phistar = phistar / (units.Gpc**3 * units.year)
        self.alpha = alpha
        self.lower_frequency = lower_frequency * units.MHz
        self.higher_frequency = higher_frequency * units.MHz
        self.cosmology = cosmology
        self.zmin = zmin
        self.zmax = zmax

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

    def __getitem__(self, name):
        return self.observations[name]

    @cached_property
    def pixels(self):
        """ """
        return numpy.arange(self.npix)

    @cached_property
    def gcrs(self):
        """ """
        pixels = numpy.arange(self.npix)
        ra, dec = self.healpix_to_lonlat(pixels)
        return coordinates.GCRS(ra=ra, dec=dec)

    @cached_property
    def itrs(self):
        """ """
        frame = coordinates.ITRS(obstime=self.itrs_time)
        return self.gcrs.transform_to(frame)

    @cached_property
    def icrs(self):
        """ """
        frame = coordinates.ICRS()
        return self.gcrs.transform_to(frame)

    @cached_property
    def xyz(self):
        """ """
        return self.itrs.cartesian.xyz

    @cached_property
    def itrs_time(self):
        """ """
        j2000 = Time('J2000').to_datetime()
        return Time(j2000)

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

    def __observe(self, telescope, location=None, name=None, radius_factor=1):

        if 'observations' not in self.__dict__:
            self.observations = {}

        noise = telescope.noise()

        location = telescope.location if location is None else location
        altaz = self.altaz(location)

        az = telescope.az
        alt = telescope.alt
        radius = radius_factor * telescope.radius

        sight = coordinates.AltAz(alt=alt, az=az, obstime=self.itrs_time)
        sight = sight.cartesian.xyz[:, numpy.newaxis]

        altaz = self.altaz(telescope.location)
        xyz = altaz.cartesian.xyz[..., numpy.newaxis]
        cosines = (xyz * sight).sum(0)
        arcs = numpy.arccos(cosines).to(units.deg)
        mask = (arcs < radius).sum(-1) > 0
        response = numpy.zeros((self.npix, telescope.n_beam))
        response[mask] = telescope.response(altaz[mask])

        array = telescope.array
        frequency_bands = telescope.frequency_bands
        sampling_time = telescope.sampling_time

        if array is not None:
            xyz = altaz.cartesian.xyz[:2]
            time_array = array @ xyz / constants.c
            time_array = time_array.to(units.ms)
        else:
            time_array = None

        observation = Observation(response, noise, frequency_bands,
                                  sampling_time, altaz, time_array)

        observation.pix = numpy.flatnonzero(mask)

        n_obs = len(self.observations)
        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        self.observations[obs_name] = observation

    def observe(self, telescopes, name=None, location=None,
                radius_factor=1):
        """

        Parameters
        ----------
        telescopes :

        name :
             (Default value = None)
        location :
             (Default value = None)
        radius_factor :
             (Default value = 1)

        Returns
        -------

        """

        if type(telescopes) is dict:
            for name, telescope in telescopes.items():
                self.__observe(telescope, location, name, radius_factor)
        else:
            self.__observe(telescopes, location, name, radius_factor)

    def __pattern(self, name, channels=False, spectral_index=0.0):

        observation = self[name]
        return observation.pattern(spectral_index, channels)

    def __response(self, name, channels=False, spectral_index=0.0):

        observation = self[name]
        si = numpy.full(self.npix, spectral_index)
        return observation.get_response(si, channels)

    def __signal(self, name, channels=False, spectral_index=0.0):

        response = self.__response(name, channels, spectral_index)
        sip = 1 + spectral_index

        nu_low = self.lower_frequency / units.MHz
        nu_high = self.higher_frequency / units.MHz

        si_factor = nu_high**sip - nu_low**sip
        signal = response / si_factor

        return numpy.abs(signal)

    def __noise(self, name, channels=False):

        observation = self[name]
        return observation.get_noise(channels)

    @numpy.errstate(divide='ignore', over='ignore')
    def __sensitivity(self, name, channels=False, spectral_index=0.0,
                      total=False):

        noise = self.__noise(name, channels)
        signal = self.__signal(name, channels, spectral_index)
        sensitivity = noise / signal

        if total:
            axes = range(1, sensitivity.ndim - int(channels))
            sensitivity = numpy.apply_over_axes(numpy.min, sensitivity, axes)

        return numpy.ma.masked_invalid(sensitivity)

    @numpy.errstate(divide='ignore', over='ignore')
    def __specific_rate(self, smin, smax, rate_unit):

        log_min = units.LogQuantity(smin)
        log_max = units.LogQuantity(smax)
        sgrid = numpy.logspace(log_min, log_max, 1000)

        zmin = self.__zdist.zmin
        zmax = self.__zdist.zmax
        zgrid = numpy.linspace(zmin, zmax, 200)

        lum_dist = self.__cosmology.luminosity_distance(zgrid)
        Lthre = 4 * numpy.pi * lum_dist[:, numpy.newaxis]**2 * sgrid

        xthre = Lthre.to(self.log_Lstar.unit) - self.log_Lstar
        xthre = xthre.to(1).value.clip(self.__lumdist.xmin,
                                       self.__lumdist.xmax)

        norm = self.phistar / self.__lumdist.pdf_norm
        lum_integral = norm * self.__lumdist.sf(xthre).clip(0.0, 1.0)
        dz = self.__zdist.angular_density(zgrid)

        integrand = (dz[:, numpy.newaxis] * lum_integral)
        zintegral = numpy.trapz(x=zgrid, y=integrand, axis=0)
        zintegral = zintegral * self.pixel_area

        xp, fp = sgrid, zintegral.to(rate_unit)

        def specific_rate(x):
            y = numpy.interp(x=x, xp=xp, fp=fp.value)
            return y * fp.unit

        return specific_rate

    @numpy.errstate(over='ignore')
    def __si_rate_map(self, name=None, channels=False, sensitivity=None,
                      SNR=None, time='day', spectral_index=0.0, total=False):

        time_unit = units.Unit(time)
        rate_unit = 1 / time_unit

        SNR = numpy.arange(1, 11) if SNR is None else SNR

        if not isinstance(sensitivity, numpy.ndarray):
            sensitivity = self.__sensitivity(name, channels, spectral_index,
                                             total)

        sensitivity = sensitivity[..., numpy.newaxis] * SNR
        sensitivity = numpy.ma.masked_invalid(sensitivity)

        smin = sensitivity.min().data
        smax = sensitivity.max().data

        specific_rate = self._HealPixMap__specific_rate(smin, smax, rate_unit)

        return specific_rate(sensitivity)

    def __rate_map(self, name=None, channels=False, sensitivity=None,
                   SNR=None, spectral_index=0.0, total=False, time='day'):

        sis = numpy.atleast_1d(spectral_index)

        rates = numpy.stack([
            self.__si_rate_map(name, channels=channels, SNR=SNR,
                               sensitivity=sensitivity, total=total,
                               spectral_index=si, time=time)
            for si in sis
        ], axis=0)

        return numpy.squeeze(rates)

    def __rate(self, name=None, channels=False, sensitivity=None,
               SNR=None, spectral_index=0.0, total=False, time='day'):

        sis = numpy.atleast_1d(spectral_index)

        rates = numpy.stack([
            self.__si_rate_map(name, channels=channels, SNR=SNR,
                               sensitivity=sensitivity, total=total,
                               spectral_index=si, time=time).sum(0)
            for si in sis
        ], axis=0)

        return numpy.squeeze(rates)

    def __response_to_noise(self, name, channels=False, spectral_index=0.0):

        response = self.__response(name, spectral_index, channels)
        noise = self.__noise(name, channels)

        return response / noise

    def __get(self, func_name=None, names=None, channels=False, **kwargs):

        func = self.__getattribute__(func_name)

        if names is None:
            names = self.observations.keys()
        elif isinstance(names, str):
            return func(names, channels, **kwargs)

        return {
            name: func(name, channels, **kwargs)
            for name in names
        }

    def rate(self, names=None, sensitivity=None, channels=False,
             SNR=None, spectral_index=0.0, total=False, time='day'):

        """

        Parameters
        ----------
        names :
             (Default value = None)
        channels :
             (Default value = False)
        SNR :
             (Default value = None)
        spectral_index :
             (Default value = 0.0)
        total :
             (Default value = False)

        Returns
        -------

        """

        if isinstance(sensitivity, numpy.ma.MaskedArray):
            rate = self.__rate(None, channels=channels, SNR=SNR,
                               sensitivity=sensitivity, total=total,
                               spectral_index=spectral_index, time=time)
            return rate

        return self.__get('_HealPixMap__rate', names, channels, SNR=SNR,
                          sensitivity=sensitivity, total=total,
                          spectral_index=spectral_index, time=time)

    def rate_map(self, names=None, sensitivity=None, channels=False,
                 SNR=None, spectral_index=0.0, total=False, time='day'):
        """

        Parameters
        ----------
        names :
             (Default value = None)
        channels :
             (Default value = False)
        SNR :
             (Default value = None)
        spectral_index :
             (Default value = 0.0)
        total :
             (Default value = False)

        Returns
        -------

        """

        if isinstance(sensitivity, numpy.ma.MaskedArray):
            rate_map = self.__rate_map(None, channels=channels, SNR=SNR,
                                       sensitivity=sensitivity, total=total,
                                       spectral_index=spectral_index,
                                       time=time)
            return rate_map

        return self.__get('_HealPixMap__rate_map', names, channels, SNR=SNR,
                          sensitivity=sensitivity, total=total,
                          spectral_index=spectral_index, time=time)

    def pattern(self, names=None, channels=False, spectral_index=0.0):
        """

        Parameters
        ----------
        names :
             (Default value = None)
        channels :
             (Default value = False)
        spectral_index :
             (Default value = 0.0)

        Returns
        -------

        """

        return self.__get('_HealPixMap__pattern', names, channels,
                          spectral_index=spectral_index)

    def response(self, names=None, channels=False, spectral_index=0.0):
        """

        Parameters
        ----------
        names :
             (Default value = None)
        channels :
             (Default value = False)
        spectral_index :
             (Default value = 0.0)

        Returns
        -------

        """

        return self.__get('_HealPixMap__response', names, channels,
                          spectral_index=spectral_index)

    def signal(self, names=None, channels=False, spectral_index=0.0):
        """

        Parameters
        ----------
        names :
             (Default value = None)
        channels :
             (Default value = False)
        spectral_index :
             (Default value = 0.0)

        Returns
        -------

        """

        return self.__get('_HealPixMap__signal', names, channels,
                          spectral_index=spectral_index)

    def noise(self, names=None, channels=False):
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

        return self.__get('_HealPixMap__noise', names, channels)

    def response_to_noise(self, names=None, channels=False,
                          spectral_index=0.0):
        """

        Parameters
        ----------
        names :
             (Default value = None)
        channels :
             (Default value = False)
        spectral_index :
             (Default value = 0.0)

        Returns
        -------

        """

        return self.__get('_HealPixMap__response_to_noise', names, channels,
                          spectral_index=spectral_index)

    def sensitivity(self, names=None, channels=False,
                    spectral_index=0.0, total=False):
        """

        Parameters
        ----------
        names :
             (Default value = None)
        channels :
             (Default value = False)
        spectral_index :
             (Default value = 0.0)
        total :
             (Default value = False)

        Returns
        -------

        """

        return self.__get('_HealPixMap__sensitivity', names, channels,
                          spectral_index=spectral_index, total=total)

    def interferometry(self, *names, time_delay=True):
        """

        Parameters
        ----------
        *names :

        time_delay :
             (Default value = True)

        Returns
        -------

        """

        observations = [self[name] for name in names]
        n_scopes = numpy.sum([
            obs.time_array.shape[0] if hasattr(obs, 'time_array') else 1
            for obs in observations
        ]).sum()

        if n_scopes > 1:

            key = '_'.join(names)
            key = 'INTF_{}'.format(key)
            interferometry = Interferometry(*observations,
                                            time_delay=time_delay)
            self.observations[key] = interferometry
        else:
            warnings.warn('Self interferometry will not be computed.')
