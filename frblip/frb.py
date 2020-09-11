import sys

import numpy

import pandas

from scipy.integrate import quad, cumtrapz

from astropy.time import Time

from astropy import cosmology, coordinates, units

from .utils import *

from .dispersion import *


class FRB():

    """
    Class which defines a Fast Radio Burst population
    """

    def __init__(self, ngen=None, days=1, zmax=6, Lstar=2.9e44,
                 L_0=9.1e41, phistar=339, alpha=-1.79,
                 wint=(.13, .33), ra=(0, 24), dec=(-90, 90),
                 si=(-15, 15), cosmo=None, verbose=True,
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

        if cosmo is None:

            self.cosmo = cosmology.Planck18_arXiv_v2

        if not verbose:

            sys.stdout = open(os.devnull, 'w')

        print("Computing the FRB rate ...")

        self.zmax = zmax
        self.L_0 = L_0 * units.erg / units.s
        self.Lstar = Lstar * units.erg / units.s
        self.phistar = phistar / (units.Gpc**3 * units.year)
        self.alpha = alpha

        self.ra_range = numpy.array(ra) * units.hourangle
        self.dec_range = numpy.array(dec) * units.degree

        self.sky_area = (4 * numpy.pi * units.sr).to(units.degree**2)

        self.sky_rate = self._sky_rate(self.zmax, self.L_0, self.Lstar,
                                       self.phistar, self.alpha, self.cosmo)

        if ra != (0, 24) or dec != (-90, 90):

            print(
                'The FoV is restricted between',
                '{} < ra < {} and {} < dec < {}.'.format(*ra, *dec),
                '\nMake sure that the survey is also',
                'restricted to this region.'
            )

            self.area = self._sky_area(self.ra_range, self.dec_range)
            self.rate = self.sky_rate * (self.area / self.sky_area)

        else:

            self.area = self.sky_area
            self.rate = self.sky_rate

        self.rate = int(self.rate.value) * self.rate.unit

        print('FRB rate =', self.rate)

        if ngen is None:

            self.ngen = int(self.rate.value * days)
            self.duration = days * (24 * units.hour)

        else:

            self.ngen = ngen
            self.duration = (ngen / self.rate).to(units.hour)

        print(self.ngen, 'FRBs will be simulated, the actual rate is',
              self.rate, '.\nTherefore it corrensponds to', self.duration,
              'of observation. \n')

        RA, DEC = self._coordinates(self.ngen, ra=self.ra_range,
                                    dec=self.dec_range)

        self.sky_coord = coordinates.SkyCoord(RA, DEC, frame='icrs')

        z, co = self._z_dist(zmax, self.ngen, self.cosmo)

        self.redshift = z
        self.comoving_distance = co

        zp1 = 1 + z

        self.luminosity_distance = zp1 * co

        self.luminosity = self._luminosity(self.ngen, self.L_0,
                                           self.Lstar, self.alpha)

        surface = 4 * numpy.pi * self.luminosity_distance**2

        self.flux = (self.luminosity / surface).to(units.Jy * units.MHz)

        self.pulse_width = numpy.random.lognormal(*wint, ngen) * units.ms

        self.arrived_pulse_width = zp1 * self.pulse_width

        time_ms = int(self.duration.to(units.ms).value)

        self.time = numpy.random.randint(time_ms, size=self.ngen) * units.ms

        self.lower_frequency = lower_frequency * units.MHz
        self.higher_frequency = higher_frequency * units.MHz

        self.spectral_index = numpy.random.uniform(*si, ngen)

        self._sip1 = self.spectral_index + 1

        nu_low = lower_frequency**self._sip1
        nu_high = higher_frequency**self._sip1

        self.dnu = nu_high - nu_low

        gal_DM = galactic_dispersion(self.sky_coord.galactic.l,
                                     self.sky_coord.galactic.b)
        host_DM = host_galaxy_dispersion(self.redshift)
        src_DM = source_dispersion(self.ngen)
        igm_DM = igm_dispersion(self.redshift, zmax=zmax)

        egal_DM = igm_DM + (src_DM + host_DM) / zp1

        self.galactic_dispersion = gal_DM
        self.host_dispersion = host_DM
        self.source_dispersion = src_DM
        self.igm_dispersion = igm_DM

        self.extra_galactic_dispersion = egal_DM

        self.dispersion = gal_DM + egal_DM

        sys.stdout = old_target

    def to_pandas(self):

        df = pandas.DataFrame({
            'Redshift': self.redshift,
            'Comoving Distance (Mpc)': self.comoving_distance,
            'Luminosity (erg/s)': self.luminosity,
            'Flux (Jy * MHz)': self.flux,
            'Right Ascension (hour)': self.sky_coord.ra.to(units.hourangle),
            'Declination (degree)': self.sky_coord.dec,
            'Galactic Longitude (degree)': self.sky_coord.galactic.l,
            'Galactic Latitude (degree)': self.sky_coord.galactic.b,
            'Intrinsic Pulse Width (ms)': self.pulse_width,
            'Arrived Pulse Width (ms)': self.arrived_pulse_width,
            'Galactic DM (pc/cm^3)': self.galactic_dispersion,
            'Host Galaxy DM (pc/cm^3)': self.host_dispersion,
            'Intergalactic Medium DM (pc/cm^3)': self.igm_dispersion,
            'Source DM (pc/cm^3)': self.source_dispersion,
            'Dispersion Measure (pc/cm^3)': self.dispersion,
            'Spectral Index': self.spectral_index,
            'Time (ms)': self.time
        }).sort_values('Time (ms)', ignore_index=True)

        return df

    def get_local_coordinates(self, location, start_time=None):

        self.location = location

        if start_time is None:
            self.start_time = Time.now()
        else:
            self.start_time = start_time

        date_time = self.start_time + self.time

        AltAzCoords = coordinates.AltAz(location=self.location,
                                        obstime=date_time)

        return self.sky_coord.transform_to(AltAzCoords)

    def _sky_area(self, ra_range, dec_range):

        x = numpy.sin(dec_range).diff() * units.rad
        y = ra_range.to(units.rad).diff()

        Area = x * y

        return Area[0].to(units.degree**2)

    def _sky_rate(self, zmax, L_0, Lstar, phistar, alpha, cosmo):

        r = L_0 / Lstar

        Lum, eps = phistar * quad(phi, r, numpy.inf, args=(alpha,))

        Vol = cosmo.comoving_volume(zmax)

        return (Lum * Vol).to(1/units.day)

    def _coordinates(self, size, ra=numpy.array([0, 24]) * units.hourangle,
                     dec=numpy.array([-90, 90]) * units.degree):

        sin = numpy.sin(dec)

        args = numpy.random.uniform(*sin, size)
        phi = numpy.arcsin(args)

        theta = numpy.random.uniform(*ra.value, size) * ra.unit

        return theta, (phi * units.rad).to(units.degree)

    def _z_dist(self, zmax, size, cosmo, zmin=.0):

        U = numpy.random.random(size)

        zs = numpy.linspace(zmin, zmax, 100)

        cdfz = cosmo.comoving_volume(zs).value
        codists = cosmo.comoving_distance(zs)

        zout = numpy.interp(x=U, xp=cdfz / cdfz[-1], fp=zs)
        rout = numpy.interp(x=zout, xp=zs, fp=codists)

        return zout, rout

    def _luminosity(self, size, L_0, Lstar, alpha):

        U = numpy.random.random(size)

        r = numpy.log10(L_0 / Lstar)

        x = numpy.linspace(r, numpy.log10(2), 100)

        L = x * Lstar

        logL = x + numpy.log10(Lstar.value)

        fL = phi(10**x, alpha)

        cdfL = cumtrapz(x=x, y=fL, initial=0)
        cdfL = cdfL / cdfL[-1]

        logs = numpy.interp(x=U, xp=cdfL, fp=logL)

        return (10**logs) * Lstar.unit

    def peak_density_flux(self, survey):

        num = survey.frequency_bands.value**self._sip1

        num = numpy.diff(num, axis=0)

        Speak = self.flux * num / (self.dnu * survey.band_widths)

        return Speak.T
