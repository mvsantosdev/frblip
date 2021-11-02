import numpy
from astropy_healpix import HEALPix

from astropy.time import Time
from astropy import coordinates, units, constants
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator

from operator import itemgetter
from functools import cached_property

from scipy.interpolate import interp1d

from .distributions import Redshift, Schechter
from .observation import Observation, Interferometry
from .cosmology import Cosmology, builtin


class HealPixMap(HEALPix):

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
        return numpy.arange(self.npix)

    @cached_property
    def gcrs(self):
        pixels = numpy.arange(self.npix)
        ra, dec = self.healpix_to_lonlat(pixels)
        return coordinates.GCRS(ra=ra, dec=dec)

    @cached_property
    def itrs(self):
        frame = coordinates.ITRS(obstime=self.itrs_time)
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
        j2000 = Time('J2000').to_datetime()
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
                radius_factor=1):

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

        return response / si_factor

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
            sensitivity = numpy.squeeze(sensitivity)

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
        covol = self.__cosmology.differential_comoving_volume(zgrid)
        dv = covol / (1 + zgrid)

        integrand = (dv[:, numpy.newaxis] * lum_integral)
        zintegral = numpy.trapz(x=zgrid, y=integrand, axis=0)
        zintegral = zintegral * self.pixel_area

        return sgrid, zintegral.to(rate_unit)

    def __rate(self, name, channels=False, SNR=None,
               spectral_index=0.0, total=False):

        sis = numpy.atleast_1d(spectral_index)

        rates = numpy.stack([
            self.__si_rate(name, channels=channels, SNR=SNR,
                           spectral_index=si, total=total)
            for si in sis
        ], axis=0)

        return numpy.squeeze(rates)

    @numpy.errstate(over='ignore')
    def __si_rate(self, name, channels=False, SNR=None, time='day',
                  spectral_index=0.0, total=False):

        SNR = numpy.arange(1, 11) if SNR is None else SNR
        sensitivity = self.__sensitivity(name, channels, spectral_index, total)

        time_unit = units.Unit(time)
        rate_unit = 1 / time_unit

        if sensitivity.ndim > 1:
            shape = sensitivity.shape
            npix = shape[0]
            final_shape = shape[1:]
            n_beam = numpy.prod(final_shape)
            sensitivity = sensitivity.reshape((-1, n_beam))
            rates = numpy.zeros((n_beam, SNR.size)) * rate_unit
        elif sensitivity.ndim == 1:
            rates = numpy.zeros(SNR.size) * rate_unit
            final_shape = None
            n_beam = None

        if sensitivity.mask.mean() == 1.0:
            return rates

        snr_max = SNR.max()

        smin = sensitivity.min()
        smax = numpy.ma.masked_invalid(snr_max * sensitivity).max()

        xp, fp = self.__specific_rate(smin.data, smax.data, rate_unit)

        for i, snr in enumerate(SNR):

            sens = snr * sensitivity

            if n_beam is None:
                s = sens.compressed()
                rate_map = numpy.interp(x=s, xp=xp, fp=fp)
                rates[i] = rate_map.sum()
            else:
                pixels, beams = sens.nonzero()
                for b in numpy.unique(beams):
                    idx = pixels[beams == b]
                    s = sens[idx, b].compressed()
                    rate_map = numpy.interp(x=s, xp=xp, fp=fp)
                    rates[b, i] = rate_map.sum()

        if final_shape is not None:
            rates = rates.reshape((*final_shape, -1))
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

    def sensitivity(self, names=None, channels=False,
                    spectral_index=0.0, total=False):

        return self.__get('_HealPixMap__sensitivity', names, channels,
                          spectral_index=spectral_index, total=total)

    def interferometry(self, *names, time_delay=True):

        key = '_'.join(names)
        key = 'INTF_{}'.format(key)
        observations = [self[name] for name in names]
        interferometry = Interferometry(*observations, time_delay=time_delay)
        self.observations[key] = interferometry
