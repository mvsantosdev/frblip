import sys

import numpy

from numpy import random

import pandas

from scipy.integrate import quad, cumtrapz

from astropy.time import Time

from astropy import cosmology, coordinates, units

from .utils import _all_sky_area, _DATA, schechter, rvs_from_cdf

from .dispersion import *


class CosmicBursts():

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

    def __getitem__(self, idx):

        return self.select(idx, inplace=False)

    def select(self, idx, inplace=False):

        frbs = self if inplace else CosmicBursts(verbose=False)

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
        frbs.time = self.spectral_index[idx]

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

            frbs.sky_area = self.sky_area
            frbs.sky_rate = self.sky_rate

            frbs.n_frb = self.n_frb
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
            self.rate = self.sky_rate * (self.area / self.sky_area)

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
