import warnings

import os

import json

import numpy

import pandas

from astropy.time import Time
from astropy import coordinates, units, constants

from scipy.special import j1

from scipy.interpolate import RectBivariateSpline
from scipy.integrate import cumtrapz

from .patterns import patterns

from .observed_bursts import ObservedBursts

from .utils import _DATA, angular_separation, simps


class RadioTelescope():

    """
    Class which defines a Radio Surveynp
    """

    def __init__(self, name='bingo', kind='gaussian', start_time=None,
                 az_shift=0, alt_shift=0, rotation=0, **kwargs):

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

        else:

            self.start_time = start_time

        name_ = '{}/{}.npz'.format(_DATA, name)

        if os.path.exists(name):

            input_dict = self._load_from_file(name)

        elif os.path.exists(name_):

            input_dict = self._load_from_file(name_)

        else:

            input_dict = kwargs

        self._load_params(**input_dict)
        self._load_beams(**input_dict)

        if kind == 'grid':

            self._load_grid(**input_dict)

        elif kind in ('tophat', 'bessel', 'gaussian'):

            self._set_selection(kind)

        else:

            print('Please choose a valid pattern kind')

        self.rotation = rotation * units.degree
        self.alt_shift = alt_shift * units.degree
        self.az_shift = az_shift * units.hourangle

        self.cos_rot = numpy.cos(self.rotation)
        self.sin_rot = numpy.sin(self.rotation)

    def __call__(self, frb, return_coords=False):

        coords = frb.get_local_coordinates(self.location)

        az, alt = self._shift_and_rotation(coords.az, coords.alt)

        response = self.selection(az, alt)

        time_factor = numpy.sqrt(frb.arrived_pulse_width / self.sampling_time)

        signal = frb.specific_flux(self._frequency)

        signal = response[:, :, numpy.newaxis] * signal[:, numpy.newaxis]
        signal = (signal / units.MHz).to(units.Jy)

        noise = self.minimum_temperature / self.gain.reshape(-1, 1)
        noise = noise.to(units.Jy)

        return ObservedBursts(signal.value, noise.value,
                              self.frequency_bands.value,
                              coords, self.location)

    def _shift_and_rotation(self, az, alt):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        rotated_az = az * self.cos_rot + alt * self.sin_rot
        rotated_alt = - az * self.sin_rot + alt * self.cos_rot

        shifted_az = rotated_az.reshape(-1, 1) - self.az_shift
        shifted_alt = rotated_alt.reshape(-1, 1) - self.alt_shift

        return shifted_az, shifted_alt

    def _load_params(self, frequency_bands, polarizations,
                     system_temperature, sampling_time,
                     longitude, latitude, elevation,
                     degradation_factor, **kwargs):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        self.degradation_factor = degradation_factor
        self.system_temperature = system_temperature * units.K

        self.frequency_bands = frequency_bands * units.MHz
        mid_frequency = 0.5 * (frequency_bands[1:] + frequency_bands[:-1])

        self._frequency = numpy.concatenate((frequency_bands, mid_frequency))
        self._frequency = numpy.sort(self._frequency) * units.MHz

        self.band_widths = numpy.diff(self.frequency_bands)
        self.sampling_time = sampling_time * units.ms
        self.polarizations = polarizations

        self.n_bands = len(self.band_widths)

        lon = coordinates.Angle(longitude, unit=units.degree)
        lat = coordinates.Angle(latitude, unit=units.degree)
        el = elevation * units.meter

        self.location = coordinates.EarthLocation(
            lon=lon, lat=lat,
            height=elevation
        )

        scaled_time = (self.band_widths * self.sampling_time).to(1)
        noise_scale = numpy.sqrt(polarizations * scaled_time.value)

        self.minimum_temperature = self.system_temperature / noise_scale

    def _load_beams(self, az, alt, solid_angle, reference_frequency,
                    **kwargs):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        self.az = az * units.degree
        self.alt = alt * units.degree
        self.solid_angle = solid_angle * units.degree**2
        self.reference_frequency = reference_frequency * units.MHz

        self.n_beams = len(solid_angle)

        self.reference_wavelength = (constants.c / self.reference_frequency)
        self.reference_wavelength = self.reference_wavelength.to(units.cm)

        sa = self.solid_angle.to(units.sr).value

        self.effective_area = (self.reference_wavelength**2 / sa)
        self.effective_area = self.effective_area.to(units.meter**2)

        self.gain = 0.5 * (self.effective_area / constants.k_B)
        self.gain = self.gain.to(units.K / units.Jy)

        sa_rad = self.solid_angle.to(units.sr).value

        arg = 1 - sa_rad / (2 * numpy.pi)

        radius = numpy.arccos(arg) * units.rad

        self.radius = radius.to(units.degree)

    def _load_grid(self, pattern, az_range, alt_range, **kwargs):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        self.pattern = pattern

        az_min, az_max, naz = az_range
        alt_min, alt_max, nalt = alt_range

        self.naz = int(naz)
        self.nalt = int(nalt)

        az_grid = numpy.linspace(az_min, az_max, self.naz)
        alt_grid = numpy.linspace(alt_min, alt_max, self.nalt)

        self.Fs = [
            numpy.vectorize(
                RectBivariateSpline(
                    az_grid, alt_grid, P
                ),
                otypes=[numpy.float]
            )
            for P in self.pattern
        ]

        self.az_min = az_min * units.degree
        self.az_max = az_max * units.degree
        self.alt_min = alt_min * units.degree
        self.alt_max = alt_max * units.degree

        self.selection = self._grid

    def _load_from_file(self, file):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        try:

            filename, extension = os.path.splitext(file)

            if extension == '.npz':

                data = numpy.load(file)

            elif extension == '.json':

                data = json.loads(file)

        except FileNotFoundError:

            print('File does not exist')

        return dict(
            polarizations=int(data.get('Polarizations')),
            system_temperature=float(data.get('System Temperature (K)')),
            sampling_time=float(data.get('Sampling Time (ms)')),
            reference_frequency=float(data.get('Reference Frequency (MHz)')),
            longitude=str(data.get('Longitude (degree)')),
            latitude=str(data.get('Latitude (degree)')),
            elevation=float(data.get('Elevation (meter)')),
            degradation_factor=float(data.get('Degradation Factor')),
            frequency_bands=numpy.array(data.get('Frequency Bands (MHz)')),
            pattern=numpy.array(data.get('Pattern')),
            az=numpy.array(data.get('Azimuth (degree)')),
            alt=numpy.array(data.get('Altitude (degree)')),
            solid_angle=numpy.array(data.get('Solid Angle (degree^2)')),
            gain=numpy.array(data.get('Gain (K/Jy)')),
            az_range=numpy.array(data.get('Azimuth Range (degree)')),
            alt_range=numpy.array(data.get('Altitude Range (degree)')),
        )

    def _set_selection(self, kind):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        self.pattern = patterns[kind]

        def selection(az, alt):

            arcs = angular_separation(az, alt, self.az, self.alt)

            rescaled_arc = (arcs / self.radius).to(1).value

            return self.pattern(rescaled_arc)

        self.selection = selection

    def _grid(self, az, alt):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        n_frbs = len(az)

        output = numpy.zeros((self.n_beams, n_frbs))

        az_in = (az > self.az_min) & (az < self.az_max)
        alt_in = (alt > self.alt_min) & (alt < self.alt_max)

        idx = az_in & alt_in

        if idx.any():

            output[:, idx] = numpy.row_stack([
                F(az[idx].value, alt[idx].value)
                for F in self.Fs
            ])

        return output