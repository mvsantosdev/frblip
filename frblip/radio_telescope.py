import os
import bz2
import dill
import json

import numpy

from operator import itemgetter
from functools import cached_property

from astropy import coordinates, units, constants
from astropy.coordinates.matrix_utilities import rotation_matrix

from .grid import CartesianGrid
from .pattern import FunctionalPattern

from .decorators import from_file, default_units


_ROOT = os.path.abspath(os.path.dirname(__file__))
_DATA = os.path.join(_ROOT, 'data')


class RadioTelescope(object):

    """Class which defines a Radio Surveynp"""

    NOISE_PERFORMANCE = {
        'total-power': 1,
        'switched': 2,
        'correlation': numpy.sqrt(2),
        '1bit-digital': 2.21,
        '2bit-digital': 1.58
    }

    @from_file(_DATA)
    def __init__(self, name='bingo', az=0.0, alt=90.0, lat=90.0,
                 lon=0.0, height=0.0, reference_frequency=150000.0,
                 system_temperature=70.0, receiver_type='total-power',
                 sampling_time=1.0, degradation_factor=1, polarizations=2,
                 directivity=23.9, frequency_range=(3e-3, 3e5),
                 kind='gaussian', array=None, offset=None):

        self._load_params(az, alt, lat, lon, height, reference_frequency,
                          system_temperature, receiver_type, sampling_time,
                          degradation_factor, polarizations, directivity,
                          frequency_range)

        self.kind = kind
        self.radius
        self.__set_array(array)
        self.__gain

        if offset is not None:
            self.alt_shift, self.az_shift = offset
            self.alt_shift = (self.alt_shift - 90) * units.deg
            self.az_shift = self.az_shift * units.degree

    @default_units(az='deg', alt='deg', lat='deg', lon='deg', height='m',
                   reference_frequency='MHz', system_temperature='K',
                   sampling_time='ms', directivity='dB(1 / sr)',
                   frequency_range='MHz')
    def _load_params(self, az, alt, lat, lon, height, reference_frequency,
                     system_temperature, receiver_type, sampling_time,
                     degradation_factor, polarizations, directivity,
                     frequency_range):

        self.az = az
        self.alt = alt
        self.lat = lat
        self.lon = lon
        self.height = height
        self.reference_frequency = reference_frequency
        self.system_temperature = system_temperature
        self.receiver_type = receiver_type
        self.sampling_time = sampling_time
        self.degradation_factor = degradation_factor
        self.polarizations = polarizations
        self.directivity = directivity
        self.frequency_range = frequency_range

    def __offset_response(self, altaz):
        altazoff = altaz.transform_to(self.offset)
        altaz = coordinates.AltAz(alt=altazoff.lat, az=altazoff.lon)
        return self.pattern(altaz)

    @cached_property
    def response(self):
        """ """
        if hasattr(self, 'alt_shift') and hasattr(self, 'az_shift'):
            return self.__offset_response
        return self.pattern

    @cached_property
    def pattern(self):
        """ """
        patterns = 'tophat', 'bessel', 'gaussian'
        error_msg = 'Please choose a valid pattern kind'
        assert self.kind in (*patterns, 'grid'), error_msg

        if self.kind == 'grid':
            params = self.__dict__
            keys = 'grid', 'xrange', 'yrange'
            grid_params = [params.pop(key) for key in keys]
            sky_params = itemgetter('alt', 'az')(params)
            return CartesianGrid(*grid_params, *sky_params)
        elif self.kind in patterns:
            keys = 'radius', 'alt', 'az', 'kind'
            pattern_params = itemgetter(*keys)(self.__dict__)
            return FunctionalPattern(*pattern_params)

    @cached_property
    def offset(self):
        """ """

        Roty = rotation_matrix(self.alt_shift, 'y')
        Rotz = rotation_matrix(-self.az_shift, 'z')
        Rot = Rotz @ Roty

        us = coordinates.UnitSphericalRepresentation(lon=self.az,
                                                     lat=self.alt)
        xyz = us.to_cartesian().xyz
        x, y, z = Rot @ xyz
        r, lat, lon = coordinates.cartesian_to_spherical(x, y, z)
        self.az = lon.to(units.deg)
        self.alt = lat.to(units.deg)

        altaz = coordinates.AltAz(alt=self.alt_shift, az=self.az_shift)
        self.offset = coordinates.SkyOffsetFrame(origin=altaz)

    def __set_array(self, array):

        if array is None:
            self.time_array = self.__none_array
        else:

            shape = array.shape
            positions, dims = shape
            assert dims == 2, "array is not a bidimensional numpy.ndarray."
            if (positions > 1) and (self.radius.size == 1):
                self.radius = numpy.tile(self.radius, positions)

            if hasattr(array, 'unit'):
                self.array = array
            else:
                self.array = array * units.m

            self.time_array = self.__array

    @cached_property
    def xyz(self):
        """ """
        loc = self.location.get_itrs()
        return loc.cartesian.xyz

    @cached_property
    def band_width(self):
        """ """
        return numpy.diff(self.frequency_range)

    @cached_property
    def minimum_temperature(self):
        """ """
        scaled_time = self.band_width * self.sampling_time
        noise_scale = numpy.sqrt(self.polarizations * scaled_time)
        return self.system_temperature / noise_scale

    @cached_property
    def noise(self):
        """ """
        noise = self.minimum_temperature / self.__gain
        return self.noise_performance * noise.to(units.Jy)

    @cached_property
    def location(self):
        """ """

        lon = self.__dict__.pop('lon')
        lat = self.__dict__.pop('lat')
        height = self.__dict__.pop('height')

        coords = dict(lon=lon, lat=lat, height=height)
        return coordinates.EarthLocation(**coords)

    @cached_property
    def noise_performance(self):
        """ """
        return RadioTelescope.NOISE_PERFORMANCE[self.receiver_type]

    @cached_property
    def solid_angle(self):
        """ """
        directivity = self.directivity.to(1 / units.sr)
        solid_angle = 4 * numpy.pi / directivity
        return solid_angle.to(units.deg**2)

    @cached_property
    def reference_wavelength(self):
        """ """
        return (constants.c / self.reference_frequency).to(units.cm)

    @cached_property
    def effective_area(self):
        """ """
        wl = self.reference_wavelength
        sa = self.solid_angle.to(units.sr).value
        return (wl**2 / sa).to(units.meter**2)

    @cached_property
    def gain(self):
        """ """
        gain = self.effective_area / constants.k_B
        return gain.to(units.K / units.Jy) / 2

    @cached_property
    def radius(self):
        """ """
        arg = 1 - self.solid_angle / (2 * numpy.pi * units.sr)
        radius = numpy.arccos(arg).to(units.deg)
        return numpy.atleast_1d(radius)

    @cached_property
    def __gain(self):
        shape = self.pattern.beams
        value = numpy.full(shape, self.gain.value)
        return value * self.gain.unit

    def __getitem__(self, idx):

        copy = dill.copy(self)

        size = self._RadioTelescope__gain.size

        copy.__dict__.update({
            attr: value[idx]
            for attr, value in copy.__dict__.items()
            if numpy.size(value) == size
        })

        return copy

    def to_pkl(self, name):
        """

        Parameters
        ----------
        name :


        Returns
        -------

        """

        output_dict = {
            attr: getattr(self, attr)
            for attr in RadioTelescope.DEFAULT_VALUES
        }

        file_name = '{}.pkl'.format(name)
        file = bz2.BZ2File(file_name, 'wb')
        dill.dump(output_dict, file, dill.HIGHEST_PROTOCOL)
        file.close()

    def to_json(self, name):
        """

        Parameters
        ----------
        name :


        Returns
        -------

        """

        output_dict = {
            attr: getattr(self, attr)
            for attr in RadioTelescope.DEFAULT_VALUES
        }

        output_dict.update({
            key: value.value
            for key, value in output_dict.items()
            if hasattr(value, 'unit')
        })

        output_dict.update({
            key: value.tolist() if value.size > 1 else value.item()
            for key, value in output_dict.items()
            if isinstance(value, numpy.ndarray)
        })

        file_name = '{}.json'.format(name)
        file = open(file_name, 'w', encoding='utf-8')
        json.dump(output_dict, file, ensure_ascii=False, indent=4)
        file.close()

    def __none_array(self, altaz):
        return None

    def __array(self, altaz):
        xy = altaz.cartesian.xyz[:2]
        path_length = self.array @ xy
        time_array = path_length.T / constants.c
        return time_array.to(units.ms)
