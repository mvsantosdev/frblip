import os

import bz2
import dill

import numpy

from operator import itemgetter
from functools import cached_property

from astropy import coordinates, units, constants
from astropy.coordinates.matrix_utilities import rotation_matrix

from .grid import CartesianGrid
from .pattern import FunctionalPattern


_ROOT = os.path.abspath(os.path.dirname(__file__))
_DATA = os.path.join(_ROOT, 'data')


noise_performance = {
    'total-power': 1,
    'switched': 2,
    'correlation': numpy.sqrt(2),
    '1bit-digital': 2.21,
    '2bit-digital': 1.58
}


class RadioTelescope(object):

    """Class which defines a Radio Surveynp"""

    DEFAULT_VALUES = {
        'az': 0.0 * units.deg,
        'alt': 90 * units.deg,
        'lat': 90 * units.deg,
        'lon': 0.0 * units.deg,
        'height': 0.0 * units.m,
        'reference_frequency': 1.5e5 * units.MHz,
        'system_temperature': 70. * units.k,
        'receiver_type': 'total-power',
        'sampling_time': 1. * units.ms,
        'degradation_factor': 1,
        'polarizations': 2,
        'directivity': 20.0 * units.LogUnit(1 / units.sr),
        'frequency_range': numpy.array([3e-3, 3e5]) * units.MHz
    }

    def __init__(self, name='bingo', kind='gaussian', array=None,
                 offset=None, location=None, **kwargs):

        """
        Creates a RadioTelescope object.

        Parameters
        ----------
        name : str
            File where the telescope parameters are stored.
            default : 'bingo'
        kind : {'tophat', 'gaussian', 'bessel', 'grid'}
            The kind of the beam pattern.
            default : 'gaussian'
        array : numpy.ndarray
            default : None
        offset : (float, float)
            default : None
        location :
            default : None
        kwargs :
            possible keys: az, alt, lat, lon, height,
            reference_frequency, system_temperature,
            receiver_type, sampling_time, degradation_factor,
            polarizations, directivity, frequency_range

        Returns
        -------
        out: RadioTelescope object.

        """

        file_name = '{}/{}.pkl'.format(_DATA, name)

        if os.path.exists(file_name):
            file = bz2.BZ2File(file_name, 'rb')
            input_dict = dill.load(file)
            file.close()
        elif os.path.exists(name):
            file = bz2.BZ2File(name, 'rb')
            input_dict = dill.load(file)
            file.close()
        else:
            input_dict = {}

        input_dict.update(kwargs)
        input_dict.update({
            key: RadioTelescope.DEFAULT_VALUES[key]
            for key in RadioTelescope.DEFAULT_VALUES
            if key not in input_dict
        })

        self.__dict__.update(input_dict)

        self.kind = kind
        self.radius
        self.__set_array(array)
        self.__gain

        if offset is not None:
            self.alt_shift, self.az_shift = offset
            self.alt_shift = (self.alt_shift - 90) * units.deg
            self.az_shift = self.az_shift * units.degree

    def __offset_response(self, altaz):
        altazoff = altaz.transform_to(self.offset)
        altaz = coordinates.AltAz(alt=altazoff.lat, az=altazoff.lon)
        return self.pattern(altaz)

    @cached_property
    def response(self):
        if hasattr(self, 'alt_shift') and hasattr(self, 'az_shift'):
            return self.__offset_response
        return self.pattern

    @cached_property
    def pattern(self):
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
        return numpy.diff(self.frequency_range)

    @cached_property
    def minimum_temperature(self):
        scaled_time = self.band_width * self.sampling_time
        noise_scale = numpy.sqrt(self.polarizations * scaled_time)
        return self.system_temperature / noise_scale

    @cached_property
    def noise(self):
        noise = self.minimum_temperature / self.__gain
        return self.noise_performance * noise.to(units.Jy)

    @cached_property
    def location(self):

        lon = self.__dict__.pop('lon')
        lat = self.__dict__.pop('lat')
        height = self.__dict__.pop('height')

        coords = dict(lon=lon, lat=lat, height=height)
        return coordinates.EarthLocation(**coords)

    @cached_property
    def noise_performance(self):
        return noise_performance[self.receiver_type]

    @cached_property
    def solid_angle(self):
        directivity = self.directivity.to(1 / units.sr)
        solid_angle = 4 * numpy.pi / directivity
        return solid_angle.to(units.deg**2)

    @cached_property
    def reference_wavelength(self):
        return (constants.c / self.reference_frequency).to(units.cm)

    @cached_property
    def effective_area(self):
        wl = self.reference_wavelength
        sa = self.solid_angle.to(units.sr).value
        return (wl**2 / sa).to(units.meter**2)

    @cached_property
    def gain(self):
        gain = self.effective_area / constants.k_B
        return gain.to(units.K / units.Jy) / 2

    @cached_property
    def radius(self):
        arg = 1 - self.solid_angle / (2 * numpy.pi * units.sr)
        radius = numpy.arccos(arg).to(units.deg)
        return numpy.atleast_1d(radius)

    @cached_property
    def __gain(self):
        shape = self.pattern.beams
        value = numpy.full(shape, self.gain.value)
        return value * self.gain.unit

    def __none_array(self, altaz):
        shape = altaz.size, self.pattern.beams
        time_array = numpy.zeros(shape)
        return time_array * units.ms

    def __array(self, altaz):
        xy = altaz.cartesian.xyz[:2]
        path_length = self.array @ xy
        time_array = path_length.T / constants.c
        return time_array.to(units.ms)
