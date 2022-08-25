import numpy
import xarray
from sparse import COO
from astropy_healpix import HEALPix

from astropy.time import Time

from functools import cached_property

from .random import Redshift, Schechter
from .cosmology import Cosmology, builtin

from astropy import units, coordinates

from .basic_sampler import BasicSampler


class HealPixMap(BasicSampler, HEALPix):
    """ """

    def __init__(self, nside=128, order='ring', phistar=339,
                 alpha=-1.79, log_Lstar=44.46, log_L0=41.96,
                 low_frequency=10, high_frequency=10000,
                 low_frequency_cal=400, high_frequency_cal=1400,
                 emission_frame=False, cosmology='Planck_18',
                 zmin=0.0, zmax=30.0):

        HEALPix.__init__(self, nside, order)

        self.__load_params(phistar, alpha, log_Lstar, log_L0,
                           low_frequency, high_frequency,
                           low_frequency_cal, high_frequency_cal,
                           emission_frame, cosmology, zmin, zmax)

        self.kind = 'PIXEL'

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

    @property
    def size(self):
        """ """
        return self.npix

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

    def luminosity_threshold(self, redshift, density_flux, spectral_index):
        """

        Parameters
        ----------
        redshift :

        density_flux :

        spectral_index :


        Returns
        -------

        """

        sip = 1 + spectral_index
        sign = numpy.sign(sip)
        nuhp = (self.high_frequency_cal / units.MHz)**sip
        nulp = (self.low_frequency_cal / units.MHz)**sip
        dnu = sign * (nuhp - nulp)

        lum_dist = self.__cosmology.luminosity_distance(redshift.ravel())
        lum_dist = lum_dist.reshape(*redshift.shape)

        Lthre = 4 * numpy.pi * dnu * lum_dist**2 * density_flux
        if self.emission_frame:
            return Lthre / (1 + redshift)**sip
        return Lthre

    def get_redshift_table(self, sensitivity, zmin=0, zmax=30,
                           spectral_index=0.0, total=False, channels=1,
                           unit='1/year', eps=1e-4):
        """

        Parameters
        ----------
        sensitivity :

        zmin :
             (Default value = 0)
        zmax :
             (Default value = 30)
        spectral_index :
             (Default value = 0.0)
        total :
             (Default value = False)
        channels :
             (Default value = 1)
        unit :
             (Default value = '1/year')
        eps :
             (Default value = 1e-4)

        Returns
        -------

        """

        data = sensitivity.data
        if not isinstance(data, COO):
            data = COO(data, fill_value=numpy.inf)
        sflux = data.data * sensitivity.unit

        smin = sflux.min()
        smax = sflux.max()

        zg, sg, table = self.get_rate_table(smin, smax, zmin, zmax,
                                            spectral_index, eps)

        table = (table * self.pixel_area).to(unit)

        rates = numpy.apply_along_axis(
            lambda x: numpy.interp(x=sflux, xp=sg, fp=x).sum(),
            axis=1, arr=table
        )

        return zg.ravel(), rates

    def get_redshift_rate(self, sensitivity, zmin=0, zmax=30,
                          spectral_index=0.0, total=False, channels=1,
                          unit='1/year', eps=1e-4):
        """

        Parameters
        ----------
        sensitivity :

        zmin :
             (Default value = 0)
        zmax :
             (Default value = 30)
        spectral_index :
             (Default value = 0.0)
        total :
             (Default value = False)
        channels :
             (Default value = 1)
        unit :
             (Default value = '1/year')
        eps :
             (Default value = 1e-4)

        Returns
        -------

        """

        zg, rates = self.get_redshift_table(sensitivity, zmin, zmax,
                                            spectral_index, total,
                                            channels, unit, eps)

        def redshift_rate(z):
            """

            Parameters
            ----------
            z :


            Returns
            -------

            """
            y = numpy.interp(x=z, xp=zg, fp=rates.value)
            return y * rates.unit

        return redshift_rate

    @numpy.errstate(divide='ignore', over='ignore')
    def get_maximum_redshift(self, sensitivity, zmin=0, zmax=30,
                             spectral_index=0.0, total=False, channels=1,
                             time=1*units.year, tolerance=1, eps=1e-4):
        """

        Parameters
        ----------
        sensitivity :

        zmin :
             (Default value = 0)
        zmax :
             (Default value = 30)
        spectral_index :
             (Default value = 0.0)
        total :
             (Default value = False)
        channels :
             (Default value = 1)
        time :
             (Default value = 1*units.year)
        tolerance :
             (Default value = 1)
        eps :
             (Default value = 1e-4)

        Returns
        -------

        """

        redshift, rates = self.get_redshift_table(sensitivity, zmin, zmax,
                                                  spectral_index, total,
                                                  channels, eps=eps)

        idx = rates.argmax()
        x = redshift[idx:][::-1]
        y = (rates[idx:][::-1] * time).to(1).clip(0)

        log_y = numpy.log(y / tolerance)
        idx = numpy.isfinite(log_y)
        center = numpy.interp(x=0, xp=log_y[idx], fp=x[idx])

        ly = y - 1.96 * numpy.sqrt(y)
        log_ly = numpy.log(ly / tolerance)
        idx = numpy.isfinite(log_ly)
        lower = numpy.interp(x=0, xp=log_ly[idx], fp=x[idx])

        uy = y + 1.96 * numpy.sqrt(y)
        log_uy = numpy.log(uy / tolerance)
        idx = numpy.isfinite(log_uy)
        upper = numpy.interp(x=0, xp=log_uy[idx], fp=x[idx])

        return center, lower, upper

    def _maximum_redshift(self, name, zmin=0, zmax=30, spectral_index=0.0,
                          snr=1, total=False, channels=1, time=1*units.year,
                          tolerance=1, eps=1e-4):
        """

        Parameters
        ----------
        name :

        zmin :
             (Default value = 0)
        zmax :
             (Default value = 30)
        spectral_index :
             (Default value = 0.0)
        snr :
             (Default value = 1)
        total :
             (Default value = False)
        channels :
             (Default value = 1)
        time :
             (Default value = 1*units.year)
        tolerance :
             (Default value = 1)
        eps :
             (Default value = 1e-4)

        Returns
        -------

        """

        sensitivity = self._sensitivity(name, spectral_index,
                                        total, channels)
        sens = snr * sensitivity
        sens.attrs['unit'] = sensitivity.unit

        zmin, zmax = self._redshift_range(name)

        return self.get_maximum_redshift(sens, zmin, zmax, spectral_index,
                                         total, channels, time, tolerance,
                                         eps)

    def _redshift_table(self, name, spectral_index=0.0, snr=1,
                        total=False, channels=1, unit='1/year',
                        eps=1e-4):
        """

        Parameters
        ----------
        name : str

        spectral_index : float
             (Default value = 0.0)
        snr : float
             (Default value = 1)
        total : bool
             (Default value = False)
        channels : int
             (Default value = 1)
        unit : astropy.units.unit or str
             (Default value = '1/year')
        eps : float
             (Default value = 1e-4)

        Returns
        -------

        """

        sensitivity = self._sensitivity(name, spectral_index,
                                        total, channels)
        sens = snr * sensitivity
        sens.attrs['unit'] = sensitivity.unit

        zmin, zmax = self._redshift_range(name)

        return self.get_redshift_table(sens, zmin, zmax, spectral_index,
                                       total, channels, unit, eps)

    def _redshift_rate(self, name, spectral_index=0.0, snr=1,
                       total=False, channels=1, unit='1/year',
                       eps=1e-4):
        """

        Parameters
        ----------
        name : str

        spectral_index : float
             (Default value = 0.0)
        snr : float
             (Default value = 1)
        total : bool
             (Default value = False)
        channels : int
             (Default value = 1)
        unit : astropy.units.unit or str
             (Default value = '1/year')
        eps : float
             (Default value = 1e-4)

        Returns
        -------

        """

        sensitivity = self._sensitivity(name, spectral_index,
                                        total, channels)
        sens = snr * sensitivity
        sens.attrs['unit'] = sensitivity.unit

        zmin, zmax = self._redshift_range(name)

        return self.get_redshift_rate(sens, zmin, zmax, spectral_index,
                                      total, channels, unit, eps)

    @numpy.errstate(divide='ignore', over='ignore')
    def get_rate_table(self, smin, smax, zmin, zmax,
                       spectral_index=0.0, eps=1e-4):
        """

        Parameters
        ----------
        smin :
             minimum sensitivity (Jy MHz)
        smax :
             minimum sensitivity (Jy MHz)
        zmin : float
             minimum redshift
        zmax : float
             maximum redshift
        unit : astropy.units.unit or str
             (Default value = '1/year')
        spectral_index : float
             spectral index value
        eps : float
             (Default value = 1e-4)

        Returns
        -------

        """

        log_min = units.LogQuantity(smin)
        log_max = units.LogQuantity(smax)

        ngrid = int(1 / eps)
        sgrid = numpy.logspace(log_min, log_max, ngrid)
        zgrid = numpy.linspace(zmin, zmax, ngrid)

        dz = self.__zdist.angular_density(zgrid)

        zgrid = zgrid.reshape(-1, 1)
        dz = dz.reshape(-1, 1)

        Lthre = self.luminosity_threshold(zgrid, sgrid, spectral_index)
        xthre = Lthre.to(self.log_Lstar.unit) - self.log_Lstar
        xmin, xmax = self.__lumdist.xmin, self.__lumdist.xmax
        xthre = xthre.to(1).value.clip(xmin, xmax)

        norm = self.phistar / self.__lumdist.pdf_norm
        lum_integral = norm * self.__lumdist.sf(xthre)

        pdf = lum_integral * dz

        return zgrid, sgrid, pdf

    def specific_rate(self, smin, smax, zmin, zmax, unit,
                      spectral_index, eps=1e-4):
        """

        Parameters
        ----------
        smin :

        smax :

        zmin :

        zmax :

        unit :

        spectral_index :

        eps :
             (Default value = 1e-4)

        Returns
        -------

        """

        zg, sg, pdf = self.get_rate_table(smin, smax, zmin, zmax,
                                          spectral_index, eps)
        spdf = numpy.trapz(x=zg, y=pdf, axis=0)
        spdf = spdf * self.pixel_area

        xp, fp = sg, spdf.to(unit)

        def specific_rate(x):
            """

            Parameters
            ----------
            x :


            Returns
            -------

            """
            y = numpy.interp(x=x, xp=xp, fp=fp.value)
            return y * fp.unit

        return specific_rate

    def _redshift_range(self, name, channels=1):
        """

        Parameters
        ----------
        name :

        channels :
             (Default value = 1)

        Returns
        -------

        """

        observation = self[name]
        return observation.redshift_range(self.low_frequency,
                                          self.high_frequency)

    def _noise(self, name, total=False, channels=1):
        """

        Parameters
        ----------
        name :

        total :
             (Default value = False)
        channels :
             (Default value = 1)

        Returns
        -------

        """

        observation = self[name]
        return observation.get_noise(total, channels)

    def _sensitivity(self, name, spectral_index=0.0, total=False, channels=1):
        """

        Parameters
        ----------
        name :

        spectral_index :
             (Default value = 0.0)
        total :
             (Default value = False)
        channels :
             (Default value = 1)

        Returns
        -------

        """

        observation = self[name]
        sign = numpy.sign(spectral_index + 1)
        spec_idx = numpy.full(self.size, spectral_index)
        freq_resp = observation.get_frequency_response(spec_idx, channels)
        noise = self._noise(name, total, channels)

        sensitivity = sign * noise / freq_resp
        sensitivity.attrs['unit'] = noise.unit / freq_resp.unit
        return sensitivity

    @numpy.errstate(over='ignore')
    def get_rate_map(self, sensitivity, zmin=0, zmax=30,
                     spectral_index=0.0, unit='year', eps=1e-4):
        """

        Parameters
        ----------
        sensitivity :

        zmin :
             (Default value = 0)
        zmax :
             (Default value = 30)
        spectral_index :
             (Default value = 0.0)
        unit :
             (Default value = 'year')
        eps :
             (Default value = 1e-4)

        Returns
        -------

        """

        if isinstance(unit, str):
            unit = units.Unit(unit)
        elif isinstance(unit, units.Quantity):
            unit = unit.unit

        assert unit.physical_type == 'time', \
               '{} is not a time unit'.format(unit.to_string())

        rate_unit = 1 / unit

        unit = sensitivity.unit
        data = sensitivity.data
        if not isinstance(data, COO):
            data = COO(data, fill_value=numpy.inf)
        sflux = data.data * unit

        smin = sflux.min()
        smax = sflux.max()

        spec_rate = self.specific_rate(smin, smax, zmin, zmax, rate_unit,
                                       spectral_index, eps)
        rates = spec_rate(sflux)

        rate_map = COO(data.coords, rates, data.shape)
        rate_map = xarray.DataArray(rate_map, dims=sensitivity.dims)
        rate_map.attrs['unit'] = rates.unit

        return rate_map

    def get_rate(self, sensitivity, zmin=0, zmax=30,
                 spectral_index=0.0, unit='year', eps=1e-4):
        """

        Parameters
        ----------
        sensitivity :

        zmin :
             (Default value = 0)
        zmax :
             (Default value = 30)
        spectral_index :
             (Default value = 0.0)
        unit :
             (Default value = 'year')
        eps :
             (Default value = 1e-4)

        Returns
        -------

        """

        rate_map = self.get_rate_map(sensitivity, zmin, zmax,
                                     spectral_index, unit, eps)
        return rate_map.sum('PIXEL', keep_attrs=True)

    def _si_rate_map(self, name=None, spectral_index=0.0, snr=None,
                     total=False, channels=1, unit='year', eps=1e-4):
        """

        Parameters
        ----------
        name :
             (Default value = None)
        spectral_index :
             (Default value = 0.0)
        snr :
             (Default value = None)
        total :
             (Default value = False)
        channels :
             (Default value = 1)
        unit :
             (Default value = 'year')
        eps :
             (Default value = 1e-4)

        Returns
        -------

        """

        s = numpy.arange(1, 11) if snr is None else snr
        s = xarray.DataArray(numpy.atleast_1d(s), dims='SNR')

        sensitivity = self._sensitivity(name, spectral_index,
                                        total, channels)

        sens = sensitivity * s
        sens.attrs = sensitivity.attrs

        zmin, zmax = self._redshift_range(name)

        return self.get_rate_map(sens, zmin, zmax, spectral_index, unit, eps)

    def _rate_map(self, name=None, spectral_index=0.0, snr=None,
                  total=False, channels=1, unit='year', eps=1e-4):
        """

        Parameters
        ----------
        name :
             (Default value = None)
        spectral_index :
             (Default value = 0.0)
        snr :
             (Default value = None)
        total :
             (Default value = False)
        channels :
             (Default value = 1)
        unit :
             (Default value = 'year')
        eps :
             (Default value = 1e-4)

        Returns
        -------

        """

        spec_idxs = numpy.atleast_1d(spectral_index)

        rates = xarray.concat([
            self._si_rate_map(name, spectral_index, snr,
                              total, channels, unit, eps)
            for spec_idx in spec_idxs
        ], dim='Spectral Index')

        return rates.squeeze()

    def _rate(self, name=None, spectral_index=0.0, snr=None,
              total=False, channels=1, unit='year', eps=1e-4):
        """

        Parameters
        ----------
        name :
             (Default value = None)
        spectral_index :
             (Default value = 0.0)
        snr :
             (Default value = None)
        total :
             (Default value = False)
        channels :
             (Default value = 1)
        unit :
             (Default value = 'year')
        eps :
             (Default value = 1e-4)

        Returns
        -------

        """

        rate_map = self._rate_map(name, spectral_index, snr,
                                  total, channels, unit, eps)
        return rate_map.sum('PIXEL', keep_attrs=True)
