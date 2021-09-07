import numpy
from sparse import COO
from astropy_healpix import HEALPix

from astropy.time import Time
from astropy import coordinates, units, constants, cosmology

from functools import cached_property

from .distributions import Redshift, Schechter
from .observation import Observation, Interferometry


class HealPixMap(HEALPix):

    def __init__(self, nside=None, order='ring', frame=None,
                 log_Lstar=44.46, log_L0=41.96, phistar=339, alpha=-1.79,
                 lower_frequency=400, higher_frequency=1400,
                 cosmo='Planck18_arXiv_v2'):

        super().__init__(nside, order, frame)

        self.__load_params(log_Lstar, log_L0, phistar, alpha, cosmo,
                           lower_frequency, higher_frequency)

    def __load_params(self, log_Lstar, log_L0, phistar, alpha, cosmo,
                      lower_frequency, higher_frequency):

        self.itrs_time = Time.now()

        self.log_L0 = log_L0 * units.LogUnit(units.erg / units.s)
        self.log_Lstar = log_Lstar * units.LogUnit(units.erg / units.s)
        self.phistar = phistar / (units.Gpc**3 * units.year)
        self.alpha = alpha
        self.lower_frequency = lower_frequency * units.MHz
        self.higher_frequency = higher_frequency * units.MHz
        self.cosmology = cosmo

    @cached_property
    def __cosmology(self):
        return cosmology.__dict__.get(self.cosmology)

    @cached_property
    def __xmin(self):
        dlogL = self.log_L0 - self.log_Lstar
        return dlogL.to(1).value

    @cached_property
    def __zdist(self):
        return Redshift(zmin=0.0, zmax=10.0,
                        cosmology=self.__cosmology)

    @cached_property
    def __lumdist(self):
        return Schechter(self.__xmin, self.alpha)

    def __getitem__(self, name):
        return self.observations[name]

    def pixels(self, cone_az, cone_alt, cone_radius):

        inside = numpy.concatenate([
            self.cone_search_lonlat(az, alt, radius)
            for az, alt, radius in zip(cone_az, cone_alt, cone_radius)
        ])
        return numpy.unique(inside)

    def altaz(self, pixels, location):

        az, alt = self.healpix_to_lonlat(pixels)
        x, y, z = coordinates.spherical_to_cartesian(1, alt, az)
        xl, yl, zl = location.x, location.y, location.z
        R = numpy.sqrt(xl**2 + yl**2 + zl**2)
        tau = R * z / constants.c
        obstime = self.itrs_time - tau
        altaz = coordinates.AltAz(alt=alt, az=az, obstime=obstime)

        return altaz

    def __observe(self, telescope, location=None, name=None, radius_factor=1):

        if 'observations' not in self.__dict__:
            self.observations = {}

        noise = telescope.noise()

        location = telescope.location if location is None else location

        az, alt = telescope.az, telescope.alt
        radius = radius_factor * telescope.radius

        pixels = numpy.arange(self.npix)
        ipixels = self.pixels(az, alt, radius)
        altaz = self.altaz(pixels, location)

        response = numpy.zeros((self.npix, telescope.n_beam))
        response[ipixels] = telescope.response(altaz[ipixels])

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
        observation.pixels = pixels

        n_obs = len(self.observations)
        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        self.observations[obs_name] = observation

    def observe(self, telescopes, name=None, location=None,
                radius_factor=1, dtype=numpy.float16):

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
        si = numpy.atleast_1d(spectral_index)
        return observation.get_response(si, channels)

    def __signal(self, name, channels=False, spectral_index=0.0):

        response = self.__response(name, spectral_index, channels)
        sip = 1 + spectral_index

        nu_low = self.lower_frequency / units.MHz
        nu_high = self.higher_frequency / units.MHz

        si_factor = nu_high**sip - nu_low**sip

        return response / si_factor

    def __noise(self, name, channels=False):

        observation = self[name]
        return observation.get_noise(channels)

    def __sensitivity(self, name, channels=False, spectral_index=0.0):

        noise = self.__noise(name, channels)
        signal = self.__signal(name, spectral_index, channels)
        return noise / signal

    def __rate(self, name, channels=False, SNR=None, total=False):

        SNR = numpy.arange(1, 11) if SNR is None else SNR

        sens = self.__sensitivity(name, channels=False)

        if total:
            axes = range(1, sens.ndim)
            sens = numpy.apply_over_axes(numpy.min, sens, axes).ravel()

        smin = sens.min()
        smax = SNR * sens[numpy.isfinite(sens)].max()
        smax = smax[numpy.isfinite(smax)].max()

        logS = units.LogUnit(sens.unit)
        log_min = smin.to(logS)
        log_max = smax.to(logS)

        zmin = self.__zdist.zmin
        zmax = self.__zdist.zmax

        sgrid = numpy.logspace(log_min, log_max, 1000)
        z = numpy.linspace(zmin, zmax, 200)

        lum_dist = self.__cosmology.luminosity_distance(z)
        Lthre = 4 * numpy.pi * lum_dist[:, numpy.newaxis]**2 * sgrid

        xthre = Lthre.to(self.log_Lstar.unit) - self.log_Lstar
        xthre = xthre.to(1).value.clip(self.__xmin)

        norm = self.phistar / self.__lumdist.pdf_norm
        lum_integral = norm * self.__lumdist.sf(xthre).clip(0.0, 1.0)
        dv = self.__cosmology.differential_comoving_volume(z) / (1 + z)

        integrand = (dv[:, numpy.newaxis] * lum_integral)
        zintegral = numpy.trapz(x=z, y=integrand, axis=0)
        zintegral = (zintegral * self.pixel_area).to(1 / units.day)

        if sens.ndim == 2:
            npix, n_beam = sens.shape
            rates = numpy.zeros((SNR.size, n_beam)) * zintegral.unit
        elif sens.ndim == 1:
            rates = numpy.zeros(SNR.size) * zintegral.unit

        for i, snr in enumerate(SNR):

            s = snr * sens
            rate_map = numpy.zeros_like(s.value) * zintegral.unit

            isfin = numpy.isfinite(s)

            rate_map[isfin] = numpy.interp(x=s[isfin], xp=sgrid, fp=zintegral)
            rates[i] = rate_map.sum(0)

        return rates

    def __response_to_noise(self, name, channels=False, spectral_index=0.0):

        response = self.__response(name, spectral_index, channels)
        noise = self.__noise(name, channels)

        return response / noise

    def __get(self, func_name=None, names=None, channels=False, **kwargs):

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

    def rate(self, names=None, channels=False, SNR=None, total=False):

        return self.__get('_HealPixMap__rate', names, channels,
                          SNR=SNR, total=total)

    def pattern(self, names=None, channels=False, spectral_index=0.0):

        return self.__get('_HealPixMap__pattern', names, channels,
                          spectral_index=spectral_index)

    def response(self, names=None, channels=False, spectral_index=0.0):

        return self.__get('_HealPixMap__response', names, channels,
                          spectral_index=spectral_index)

    def signal(self, names=None, channels=False, spectral_index=0.0):

        return self.__get('_HealPixMap__signal', names, channels,
                          spectral_index=spectral_index)

    def noise(self, names=None, channels=False):

        return self.__get('_HealPixMap__noise', names, channels)

    def response_to_noise(self, names=None, channels=False,
                          spectral_index=0.0):

        return self.__get('_HealPixMap__response_to_noise', names, channels,
                          spectral_index=spectral_index)

    def sensitivity(self, names=None, channels=False, spectral_index=0.0):

        return self.__get('_HealPixMap__sensitivity', names, channels,
                          spectral_index=spectral_index)

    def interferometry(self, *names, time_delay=True):

        key = '_'.join(names)
        key = 'INTF_{}'.format(key)
        observations = [self[name] for name in names]
        interferometry = Interferometry(*observations, time_delay=time_delay)
        self.observations[key] = interferometry
