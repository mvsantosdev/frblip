import numpy
import xarray
from astropy_healpix import HEALPix

from astropy.time import Time
from astropy import coordinates, units, constants
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator

from operator import itemgetter
from functools import cached_property

from .random import Redshift, Schechter
from .observation import Observation, Interferometry
from .cosmology import Cosmology, builtin


class HealPixMap(HEALPix):
    """ """

    def __init__(self, nside=128, order='ring', phistar=339,
                 alpha=-1.79, log_Lstar=44.46, log_L0=41.96,
                 low_frequency=10, high_frequency=10000,
                 low_frequency_cal=400, high_frequency_cal=1400,
                 emission_frame=True, cosmology='Planck_18',
                 zmin=0.0, zmax=30.0):

        super().__init__(nside, order)

        self.__load_params(phistar, alpha, log_Lstar, log_L0,
                           low_frequency, high_frequency,
                           low_frequency_cal, high_frequency_cal,
                           emission_frame, cosmology, zmin, zmax)

    def __load_params(self, phistar, alpha, log_Lstar, log_L0,
                      low_frequency, high_frequency,
                      low_frequency_cal, high_frequency_cal,
                      emission_frame, cosmology, zmin, zmax):

        self.log_L0 = log_L0 * units.LogUnit(units.erg / units.s)
        self.log_Lstar = log_Lstar * units.LogUnit(units.erg / units.s)
        self.phistar = phistar / (units.Gpc**3 * units.year)
        self.alpha = alpha
        self.low_frequency = low_frequency * units.MHz
        self.high_frequency = high_frequency * units.MHz
        self.low_frequency_cal = low_frequency_cal * units.MHz
        self.high_frequency_cal = high_frequency_cal * units.MHz
        self.emission_frame = emission_frame
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

        n_obs = len(self.observations)
        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        location = telescope.location if location is None else location
        altaz = self.altaz(location)

        az = telescope.az
        alt = telescope.alt
        radius = radius_factor * telescope.radius

        sight = coordinates.AltAz(alt=alt, az=az, obstime=self.itrs_time)
        sight = sight.cartesian.xyz[:, numpy.newaxis]

        sampling_time = telescope.sampling_time
        frequency_range = telescope.frequency_range
        width = telescope.frequency_range.diff()

        altaz = self.altaz(telescope.location)
        xyz = altaz.cartesian.xyz[..., numpy.newaxis]
        cosines = (xyz * sight).sum(0)
        arcs = numpy.arccos(cosines).to(units.deg)
        mask = (arcs < radius).sum(-1) > 0

        resp = telescope.response(altaz[mask])
        response = numpy.zeros((self.npix, *resp.shape[1:]))
        response[mask] = resp
        response = response * (units.MHz / width).to(1)
        dims = 'PIXEL', obs_name
        response = xarray.DataArray(response, dims=dims, name='Response')
        response.name = 'Response'

        noise = telescope.noise()
        noise = (noise / units.Jy).to(1)
        noise = xarray.DataArray(noise, dims=obs_name, name='Noise')
        noise.name = 'Noise'

        time_array = telescope.time_array(altaz)
        time_array = (time_array * units.MHz).to(1)
        time_array = xarray.DataArray(time_array, dims=('FRB', obs_name))
        time_array.name = 'Time Array'

        observation = Observation(response, noise, altaz, time_array,
                                  frequency_range, sampling_time)

        observation.pix = numpy.flatnonzero(mask)

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

    def __response(self, name, channels=1, spectral_index=0.0):

        observation = self[name]
        si = numpy.full(self.npix, spectral_index)
        return observation.get_response(si, channels)

    def __noise(self, name, channels=1):

        observation = self[name]
        return observation.get_noise(channels)

    @numpy.errstate(divide='ignore', over='ignore')
    def __sensitivity(self, name, channels=1, spectral_index=0.0,
                      total=False, level=None):

        noise = self.__noise(name, channels)
        response = self.__response(name, channels, spectral_index)
        sensitivity = (1 / response) * noise

        if total:
            lvl = sensitivity.dims[1:-1]
            return sensitivity.min(lvl)
        if level is not None:
            return sensitivity.min(level)
        return sensitivity

    @numpy.errstate(divide='ignore', over='ignore')
    def __specific_rate(self, smin, smax, zmin, zmax, frequency_range,
                        rate_unit, spectral_index):

        log_min = units.LogQuantity(smin)
        log_max = units.LogQuantity(smax)
        sgrid = numpy.logspace(log_min, log_max, 1000)

        zgrid = numpy.linspace(zmin, zmax, 200)

        lum_dist = self.__cosmology.luminosity_distance(zgrid)
        dz = self.__zdist.angular_density(zgrid)

        dz = dz[:, numpy.newaxis]
        zgrid = zgrid[:, numpy.newaxis]
        lum_dist = lum_dist[:, numpy.newaxis]

        _sip = 1 + spectral_index
        nuhp = (self.high_frequency_cal / units.MHz)**_sip
        nulp = (self.low_frequency_cal / units.MHz)**_sip
        dnup = (frequency_range / units.MHz)**_sip
        width = frequency_range.diff()
        nu_factor = width * (nuhp - nulp) / dnup.diff(axis=-1)

        Lthre = 4 * numpy.pi * nu_factor * lum_dist**2 * sgrid
        if self.emission_frame:
            Lthre = Lthre / (1 + zgrid)**_sip

        xthre = Lthre.to(self.log_Lstar.unit) - self.log_Lstar
        xmin, xmax = self.__lumdist.xmin, self.__lumdist.xmax
        xthre = xthre.to(1).value.clip(xmin, xmax)

        norm = self.phistar / self.__lumdist.pdf_norm
        lum_integral = norm * self.__lumdist.sf(xthre).clip(0.0, 1.0)

        integrand = lum_integral * dz
        zintegral = numpy.trapz(x=zgrid, y=integrand, axis=0)
        zintegral = zintegral * self.pixel_area

        xp, fp = sgrid, zintegral.to(rate_unit)

        def specific_rate(x):
            y = numpy.interp(x=x, xp=xp, fp=fp.value)
            return y * fp.unit

        return specific_rate

    @numpy.errstate(over='ignore')
    def __si_rate_map(self, name=None, channels=1, sensitivity=None,
                      SNR=None, time='day', spectral_index=0.0, total=False):

        time_unit = units.Unit(time)
        rate_unit = 1 / time_unit

        S = numpy.arange(1, 11) if SNR is None else SNR
        S = xarray.DataArray(S, dims='SNR')

        if not isinstance(sensitivity, numpy.ndarray):
            sensitivity = self.__sensitivity(name, channels,
                                             spectral_index,
                                             total)

        sensitivity = sensitivity * S
        sensitivity = numpy.ma.masked_invalid(sensitivity)

        smin = sensitivity.min() * units.Jy
        smax = sensitivity.max() * units.Jy

        frequency_range = self[name].frequency_range
        zmax = self.high_frequency / frequency_range[0] - 1
        zmin = self.low_frequency / frequency_range[-1] - 1
        zmin = zmin.clip(0)

        specific_rate = self.__specific_rate(smin, smax, zmin, zmax,
                                             frequency_range, rate_unit,
                                             spectral_index)

        return specific_rate(sensitivity)

    def __rate_map(self, name=None, channels=1, sensitivity=None,
                   SNR=None, spectral_index=0.0, total=False, time='day'):

        sis = numpy.atleast_1d(spectral_index)

        rates = numpy.stack([
            self.__si_rate_map(name, channels=channels, SNR=SNR,
                               sensitivity=sensitivity, total=total,
                               spectral_index=si, time=time)
            for si in sis
        ], axis=0)

        return numpy.squeeze(rates)

    def __rate(self, name=None, channels=1, sensitivity=None,
               SNR=None, spectral_index=0.0, total=False, time='day'):

        sis = numpy.atleast_1d(spectral_index)

        rates = numpy.stack([
            self.__si_rate_map(name, channels=channels, SNR=SNR,
                               sensitivity=sensitivity, total=total,
                               spectral_index=si, time=time).sum(0)
            for si in sis
        ], axis=0)

        return numpy.squeeze(rates)

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

    def rate(self, names=None, sensitivity=None, channels=1,
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

    def rate_map(self, names=None, sensitivity=None, channels=1,
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

    def response(self, names=None, channels=1, spectral_index=0.0):
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

        return self.__get('_HealPixMap__noise', names, channels)

    def sensitivity(self, names=None, channels=1, spectral_index=0.0,
                    total=False, level=None):
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
                          spectral_index=spectral_index, total=total,
                          level=level)

    def interferometry(self, namei, namej=None):
        """

        Parameters
        ----------
        *names :

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
        interferometry = Interferometry(obsi, obsj)
        self.observations[key] = interferometry
