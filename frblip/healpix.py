import numpy
from sparse import COO
from astropy_healpix import HEALPix

from astropy.time import Time
from astropy import coordinates, units, constants, cosmology
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator

from operator import itemgetter
from functools import cached_property

from .distributions import Redshift, Schechter
from .observation import Observation, Interferometry


class HealPixMap(HEALPix):

    def __init__(self, nside=None, order='ring', frame=None,
                 cosmo='Planck18_arXiv_v2', phistar=339,
                 alpha=-1.79, log_Lstar=44.46, log_L0=41.96,
                 lower_frequency=400, higher_frequency=1400):

        super().__init__(nside, order, frame)

        self.__load_params(log_Lstar, log_L0, phistar, alpha, cosmo,
                           lower_frequency, higher_frequency)

    def __load_params(self, log_Lstar, log_L0, phistar, alpha, cosmo,
                      lower_frequency, higher_frequency):

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
        return Redshift(zmin=0.0, zmax=2.15,
                        cosmology=self.__cosmology)

    @cached_property
    def __lumdist(self):
        return Schechter(self.__xmin, self.alpha)

    def __getitem__(self, name):
        return self.observations[name]

    @cached_property
    def pixels(self):
        return numpy.arange(self.npix)

    @cached_property
    def gcrs(self):
        pixels = numpy.arange(self.npix)
        ra, dec = self.healpix_to_lonlat(pixels)
        return coordinates.GCRS(ra=ra, dec=dec)

    @cached_property
    def itrs(self):
        frame = coordinates.ITRS()
        return self.gcrs.transform_to(frame)

    @cached_property
    def icrs(self):
        frame = coordinates.ICRS()
        return self.gcrs.transform_to(frame)

    @cached_property
    def xyz(self):
        return self.itrs.cartesian.xyz

    @cached_property
    def itrs_time(self):
        j2000 = self.itrs.obstime.to_datetime()
        return Time(j2000)

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

    def __rate(self, name, channels=False, SNR=None,
               spectral_index=0.0, total=False):

        sis = numpy.atleast_1d(spectral_index)

        rates = numpy.stack([
            self.__si_rate(name, channels=channels, SNR=SNR,
                           spectral_index=si, total=total)
            for si in sis
        ], axis=0)

        return numpy.squeeze(rates)

    def __si_rate(self, name, channels=False, SNR=None,
                  spectral_index=0.0, total=False):

        SNR = numpy.arange(1, 11) if SNR is None else SNR

        sens = self.__sensitivity(name, channels=False,
                                  spectral_index=spectral_index)

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
            rates[i] = numpy.nansum(rate_map, axis=0)

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

    def rate(self, names=None, channels=False, SNR=None,
             spectral_index=0.0, total=False):

        return self.__get('_HealPixMap__rate', names, channels, SNR=SNR,
                          spectral_index=spectral_index, total=total)

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