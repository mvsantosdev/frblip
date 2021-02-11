import warnings

import os

import json

import numpy

import pandas

from astropy.time import Time
from astropy import coordinates, units, constants

from scipy.special import j1

from scipy.integrate import cumtrapz

from .grid import CartesianGrid
from .pattern import FunctionalPattern

from .utils import _all_sky_area, _DATA


class RadioTelescope(object):

    """
    Class which defines a Radio Surveynp
    """

    def __init__(self, name='bingo', kind='gaussian',
                 start_time=None, **kwargs):

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

        if os.path.exists(name_):

            input_dict = self._load_from_file(name_)

        elif os.path.exists(name):

            input_dict = self._load_from_file(name)

        else:

            input_dict = kwargs

        self._load_params(**input_dict)
        self._load_beams(**input_dict)

        self.kind = kind

        if kind == 'grid':

            self.selection = CartesianGrid(**input_dict)

        elif kind in ('tophat', 'bessel', 'gaussian'):

            self.selection = FunctionalPattern(kind, **input_dict)

        else:

            print('Please choose a valid pattern kind')

    def _load_params(self, frequency_bands, polarizations,
                     system_temperature, sampling_time,
                     longitude, latitude, height,
                     degradation_factor, **kwargs):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        self.degradation_factor = degradation_factor
        self.system_temperature = system_temperature * units.K

        self.frequency_bands = frequency_bands * units.MHz
        mid_frequency = 0.5 * (frequency_bands[1:] + frequency_bands[:-1])

        self._frequency = numpy.concatenate((frequency_bands,
                                             mid_frequency))
        self._frequency = numpy.sort(self._frequency) * units.MHz

        self.band_widths = numpy.diff(self.frequency_bands)
        self.sampling_time = sampling_time * units.ms
        self.polarizations = polarizations

        self.n_bands = len(self.band_widths)

        lon = coordinates.Angle(longitude, unit=units.degree)
        lat = coordinates.Angle(latitude, unit=units.degree)
        h = height * units.meter

        self.location = coordinates.EarthLocation(lon=lon, lat=lat,
                                                  height=h)

        scaled_time = (self.band_widths * self.sampling_time).to(1)
        noise_scale = numpy.sqrt(polarizations * scaled_time.value)

        self.minimum_temperature = self.system_temperature / noise_scale

    def _load_beams(self, az, alt, maximum_gain, reference_frequency,
                    **kwargs):

        """
        This is a private function, please do not call it
        directly unless you know exactly what you are doing.
        """

        self.az = az * units.degree
        self.alt = alt * units.degree

        self.solid_angle = _all_sky_area * 10**(- 0.1 * maximum_gain)
        self.solid_angle = self.solid_angle.to(units.degree**2)

        self.reference_frequency = reference_frequency * units.MHz

        self.n_beam = len(maximum_gain)

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
            height=float(data.get('Height (meter)')),
            degradation_factor=float(data.get('Degradation Factor')),
            frequency_bands=numpy.array(data.get('Frequency Bands (MHz)')),
            grid=numpy.array(data.get('Pattern')),
            az=numpy.array(data.get('Azimuth (degree)')),
            alt=numpy.array(data.get('Altitude (degree)')),
            maximum_gain=numpy.array(data.get('Maximum Gain (dB)')),
            xrange=numpy.array(data.get('U')),
            yrange=numpy.array(data.get('V')),
        )
