import numpy

import pandas

from astropy.time import Time
from astropy import coordinates, units

from scipy.special import j1

from scipy.interpolate import RectBivariateSpline

from .patterns import patterns

from .utils import _DATA, angular_separation


class Survey():

    """
    Class which defines a Radio Survey
    """

    def __init__(self, name='bingo', file=None, kind='gaussian',
                 az0=0.0, alt0=0.0, polarizations=2,
                 frequency_bands=[1000.0, 1300.0],
                 system_temperature=70.0, sampling_time=1.0,
                 degradation_factor=1.0, start_time=None):

        """
        Creates a Survey object.

        Parameters
        ----------
        file : str
            File where the telescope parameters are stored.
        kind : {'tophat', 'gaussian', 'bessel', 'grid'}
            The kind of the beam pattern.
        az0 : float
            Translation on Azimuth coordinate (degrees)
        alt0 : float
            Translation on Altitude coordinate (degrees)
        frequency_band : list(float)
            List which contains the frequency bands bounds (MHz).
        polarizations : int
            Number of Polarizations.
        system_temperature : float
            System temperature (K).
        sampling_time : float
            Sampling Time (ms).
        degradation_factor : float
            Degradation factor.
        start_time : str
            Survey start time ().
        name : str
            Name of the survey.
        Returns
        -------
        out: Survey object.

        """

        if start_time is None:
            self.start_time = Time.now()

        self.name = name

        self.start_time = start_time

        self.degradation_factor = degradation_factor
        self.system_temperature = system_temperature * units.K
        self.frequency_bands = numpy.array(frequency_bands).reshape(-1, 1)
        self.frequency_bands = self.frequency_bands * units.MHz
        self.band_widths = numpy.diff(self.frequency_bands, axis=0)
        self.sampling_time = sampling_time * units.ms
        self.polarizations = polarizations
        
        beams = None
        
        if file is None:
        
            beams = numpy.load('{}/{}.npz'.format(_DATA, name))
            
        else:
            
            beams = numpy.load(file)

        self.location = coordinates.EarthLocation(
            lat=coordinates.Angle(beams['Latitude (degree)'], unit=units.deg),
            lon=coordinates.Angle(beams['Longitude (degree)'], unit=units.deg),
            height=beams['Elevation (meter)'] * units.m
        )

        self.az = (beams['Azimuth (degree)'] + az0) * units.degree
        self.alt = (beams['Altitude (degree)'] + alt0) * units.degree
        self.solid_angle = beams['Solid Angle (degree^2)'] * units.degree**2
        self.gain = beams['Gain (K/Jy)'] * units.K / units.Jy

        self.nbeams = len(self.gain)

        self.columns = [
            '{}_{}'.format(self.name, i)
            for i in range(self.nbeams)
        ]

        sa_rad = self.solid_angle.to(units.sr).value

        radius = numpy.arccos(1 - sa_rad / (2 * numpy.pi)) * units.rad

        self.radius = radius.to(units.degree)

        if kind == 'grid':

            self.pattern = beams['Pattern']

            az_min, az_max, naz = beams['Azimuth Range (degree)']
            alt_min, alt_max, nalt = beams['Altitude Range (degree)']

            self.az_min, self.az_max = az_min, az_max
            self.alt_min, self.alt_max = alt_min, alt_max

            self.naz = int(naz)
            self.nalt = int(nalt)

            self.az_grid = numpy.linspace(az_min, az_max, self.naz)
            self.alt_grid = numpy.linspace(alt_min, alt_max, self.nalt)

            self.az_grid += az0
            self.alt_grid += alt0

            self.Fs = [
                numpy.vectorize(
                    RectBivariateSpline(self.az_grid, self.alt_grid, P),
                    otypes=[numpy.float]
                )
                for P in self.pattern
            ]

            self.selection = self._grid

        elif kind in ('tophat', 'bessel', 'gaussian'):

            self.pattern = patterns[kind]

            def selection(az, alt):

                arcs = angular_separation(az, alt, self.az, self.alt)

                rescaled_arc = (arcs / self.radius).to(1).value

                return self.pattern(rescaled_arc)

            self.selection = selection

        else:

            print('Please choose a valid pattern kind')

        scaled_time = self.band_widths * self.sampling_time
        w = numpy.sqrt(self.polarizations * scaled_time)

        self.minimum_temperature = (self.system_temperature / w).to(units.K)

    def _grid(self, az, alt):

        return numpy.row_stack([
            F(az.value, alt.value)
            for F in self.Fs
        ])

    def __call__(self, frb):

        coords = frb.get_local_coordinates(self.location)

        response = self.selection(coords.az, coords.alt) * self.gain

        time_factor = numpy.sqrt(frb.arrived_pulse_width / self.sampling_time)

        Speak = frb.peak_density_flux(self) * time_factor.reshape(-1, 1)

        signal = response[:, :, numpy.newaxis] * Speak

        return coords, signal.swapaxes(0, 1)
