from __future__ import annotations

import bz2
import json
import os
from functools import cached_property
from operator import itemgetter

import dill
import numpy
from astropy import constants, coordinates, units
from astropy.coordinates.matrix_utilities import rotation_matrix

from .decorators import default_units, from_source
from .grid import CartesianGrid
from .pattern import FunctionalPattern

_ROOT = os.path.abspath(os.path.dirname(__file__))
_DATA = os.path.join(_ROOT, 'data')


class RadioTelescope(object):

    NOISE_PERFORMANCE = {
        'total-power': 1,
        'switched': 2,
        'correlation': numpy.sqrt(2),
        '1bit-digital': 2.21,
        '2bit-digital': 1.58,
    }

    @from_source(_DATA)
    def __init__(
        self,
        source: str | dict | None = 'bingo',
        az: units.Quantity | float = 0.0,
        alt: units.Quantity | float = 90.0,
        lat: units.Quantity | float = 90.0,
        lon: units.Quantity | float = 0.0,
        height: units.Quantity | float = 0.0,
        reference_frequency: units.Quantity | float = 150000.0,
        system_temperature: units.Quantity | float = 70.0,
        receiver_type: str = 'total-power',
        sampling_time: units.Quantity | float = 1.0,
        degradation_factor: float = 1,
        polarizations: int = 2,
        directivity: units.Quantity | float = 23.9,
        frequency_range: units.Quantity | tuple[float, float] = (3e-3, 3e5),
        kind: str = 'gaussian',
        array: units.Quantity | numpy.ndarray | None = None,
        offset: units.Quantity | tuple[float, float] = None,
    ):

        self._load_params(
            az,
            alt,
            lat,
            lon,
            height,
            reference_frequency,
            system_temperature,
            receiver_type,
            sampling_time,
            degradation_factor,
            polarizations,
            directivity,
            frequency_range,
        )

        self.kind = kind
        self.radius
        self._set_array(array)
        self._set_offset(offset)
        self._gain

    @default_units(
        az='deg',
        alt='deg',
        lat='deg',
        lon='deg',
        height='m',
        reference_frequency='MHz',
        system_temperature='K',
        sampling_time='ms',
        directivity='dB(1 / sr)',
        frequency_range='MHz',
    )
    def _load_params(
        self,
        az: units.Quantity | float,
        alt: units.Quantity | float,
        lat: units.Quantity | float,
        lon: units.Quantity | float,
        height: units.Quantity | float,
        reference_frequency: units.Quantity | float,
        system_temperature: units.Quantity | float,
        receiver_type: str,
        sampling_time: units.Quantity | float,
        degradation_factor: float,
        polarizations: int,
        directivity: units.Quantity | float,
        frequency_range: units.Quantity | tuple[float, float],
    ):

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

    def _offset_response(self, altaz: coordinates.SkyCoord) -> numpy.ndarray:
        altazoff = altaz.transform_to(self.offset)
        altaz = coordinates.AltAz(alt=altazoff.lat, az=altazoff.lon)
        return self.pattern(altaz)

    @cached_property
    def response(self) -> FunctionalPattern:

        if hasattr(self, 'alt_shift') and hasattr(self, 'az_shift'):
            return self._offset_response
        return self.pattern

    @cached_property
    def pattern(self) -> FunctionalPattern | CartesianGrid:

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

        us = coordinates.UnitSphericalRepresentation(lon=self.az, lat=self.alt)
        cartesian = us.to_cartesian()
        x, y, z = Rot @ cartesian.xyz
        r, lat, lon = coordinates.cartesian_to_spherical(x, y, z)
        self.az = lon.to(units.deg)
        self.alt = lat.to(units.deg)

        altaz = coordinates.AltAz(alt=self.alt_shift, az=self.az_shift)
        self.offset = coordinates.SkyOffsetFrame(origin=altaz)

    @default_units(array='m')
    def _set_array(self, array: units.Quantity | numpy.ndarray | None):

        if array is None:
            self.time_array = lambda altaz: None
        else:
            shape = array.shape
            positions, dims = shape
            assert dims == 2, 'array is not bidimensional.'
            if (positions > 1) and (self.radius.size == 1):
                self.radius = numpy.tile(self.radius, positions)
            self.time_array = self._array
            self.array = array

    @default_units(offset='deg')
    def _set_offset(self, offset: units.Quantity | numpy.ndarray | None):
        if offset is not None:
            self.alt_shift, self.az_shift = offset
            self.alt_shift = self.alt_shift - 90 * units.deg
            self.az_shift = self.az_shift * units.degree

    @cached_property
    def xyz(self) -> units.Quantity:

        loc = self.location.get_itrs()
        return loc.cartesian.xyz

    @cached_property
    def band_width(self) -> units.Quantity:

        return numpy.diff(self.frequency_range)

    @cached_property
    def minimum_temperature(self) -> units.Quantity:

        scaled_time = self.band_width * self.sampling_time
        noise_scale = numpy.sqrt(self.polarizations * scaled_time)
        return self.system_temperature / noise_scale

    @cached_property
    def noise(self) -> units.Quantity:

        noise = self.minimum_temperature / self._gain
        return self.noise_performance * noise.to(units.Jy)

    @cached_property
    def location(self) -> coordinates.EarthLocation:

        lon = self.__dict__.pop('lon')
        lat = self.__dict__.pop('lat')
        height = self.__dict__.pop('height')

        coords = dict(lon=lon, lat=lat, height=height)
        return coordinates.EarthLocation(**coords)

    @cached_property
    def noise_performance(self) -> float:
        return RadioTelescope.NOISE_PERFORMANCE[self.receiver_type]

    @cached_property
    def solid_angle(self) -> units.Quantity:
        directivity = self.directivity.to(1 / units.sr)
        solid_angle = 4 * numpy.pi / directivity
        return solid_angle.to(units.deg**2)

    @cached_property
    def reference_wavelength(self) -> units.Quantity:
        return (constants.c / self.reference_frequency).to(units.cm)

    @cached_property
    def effective_area(self) -> units.Quantity:
        wl = self.reference_wavelength
        sa = self.solid_angle.to(units.sr).value
        return (wl**2 / sa).to(units.meter**2)

    @cached_property
    def gain(self) -> units.Quantity:
        gain = self.effective_area / constants.k_B
        return gain.to(units.K / units.Jy) / 2

    @cached_property
    def radius(self) -> units.Quantity:
        arg = 1 - self.solid_angle / (2 * numpy.pi * units.sr)
        radius = numpy.arccos(arg).to(units.deg)
        return numpy.atleast_1d(radius)

    @cached_property
    def _gain(self) -> units.Quantity:
        shape = self.pattern.beams
        value = numpy.full(shape, self.gain.value)
        return value * self.gain.unit

    def __getitem__(
        self,
        idx: int | str | slice | numpy.ndarray,
    ) -> RadioTelescope:

        copy = dill.copy(self)

        size = self._RadioTelescope_gain.size

        copy.__dict__.update(
            {
                attr: value[idx]
                for attr, value in copy.__dict__.items()
                if numpy.size(value) == size
            }
        )

        return copy

    def to_pkl(self, name: str):

        output_dict = {
            attr: getattr(self, attr) for attr in RadioTelescope.DEFAULT_VALUES
        }

        file_name = '{}.pkl'.format(name)
        file = bz2.BZ2File(file_name, 'wb')
        dill.dump(output_dict, file, dill.HIGHEST_PROTOCOL)
        file.close()

    def to_json(self, name: str):

        output_dict = {
            attr: getattr(self, attr) for attr in RadioTelescope.DEFAULT_VALUES
        }

        output_dict.update(
            {
                key: value.value
                for key, value in output_dict.items()
                if hasattr(value, 'unit')
            }
        )

        output_dict.update(
            {
                key: value.tolist() if value.size > 1 else value.item()
                for key, value in output_dict.items()
                if isinstance(value, numpy.ndarray)
            }
        )

        file_name = '{}.json'.format(name)
        file = open(file_name, 'w', encoding='utf-8')
        json.dump(output_dict, file, ensure_ascii=False, indent=4)
        file.close()

    def _array(self, altaz: coordinates.SkyCoord) -> units.Quantity:
        xy = altaz.cartesian.xyz[:2]
        path_length = self.array @ xy
        time_array = path_length.T / constants.c
        return time_array.to(units.ms)
