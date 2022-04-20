import os

import numpy

from functools import cached_property

from astropy import coordinates, units, constants
from astropy.coordinates.matrix_utilities import rotation_matrix

from .grid import CartesianGrid
from .pattern import FunctionalPattern

from .utils import _DATA, load_file, load_params, sub_dict


noise_performance = {
    'total-power': 1,
    'switched': 2,
    'correlation': numpy.sqrt(2),
    '1bit-digital': 2.21,
    '2bit-digital': 1.58
}


class RadioTelescope(object):

    """Class which defines a Radio Surveynp"""

    def __init__(self, name='bingo', kind='gaussian', array=None,
                 offset=None, location=None):

        """
        Creates a RadioTelescope object.

        Parameters
        ----------
        name : str
            File where the telescope parameters are stored.
        kind : {'tophat', 'gaussian', 'bessel', 'grid'}
            The kind of the beam pattern.
        start_time : str
            Survey start time ().

        Returns
        -------
        out: RadioTelescope object.

        """

        name_ = '{}/{}.npz'.format(_DATA, name)
        name_ = name_ if os.path.exists(name_) else name
        input_dict = load_file(name_)
        input_dict = load_params(input_dict)

        self.__dict__.update(input_dict)

        self.kind = kind
        keys = ['grid', 'xrange', 'yrange']
        grid_params = sub_dict(input_dict, keys=keys, pop=True)

        if kind == 'grid':
            coords = sub_dict(input_dict, keys=['alt', 'az'])
            grid_params.update(coords)
            self.__response = CartesianGrid(**grid_params)
        elif kind in ('tophat', 'bessel', 'gaussian'):
            keys = ['kind', 'directivity', 'alt', 'az']
            pattern_params = sub_dict(input_dict, keys=keys)
            self.__response = FunctionalPattern(**pattern_params)
        else:
            print('Please choose a valid pattern kind')

        if hasattr(array, 'unit'):
            self.array = array
        elif array is None:
            self.array = array
        else:
            self.array = array * units.m

        self.__derived()

        if offset is not None:
            self.set_offset(*offset)
        else:
            self.response = self.__no_offset_response

        if location is not None:
            self.set_location(location)

    @cached_property
    def xyz(self):
        """ """
        loc = self.location.get_itrs()
        return loc.cartesian.xyz

    def set_offset(self, alt, az):
        """

        Parameters
        ----------
        alt :

        az :


        Returns
        -------

        """

        alt_shift = (alt - 90) * units.deg
        az_shift = az * units.degree

        Roty = rotation_matrix(alt_shift, 'y')
        Rotz = rotation_matrix(-az_shift, 'z')
        Rot = Rotz @ Roty

        us = coordinates.UnitSphericalRepresentation(lon=self.az,
                                                     lat=self.alt)
        xyz = us.to_cartesian().xyz
        x, y, z = Rot @ xyz
        r, lat, lon = coordinates.cartesian_to_spherical(x, y, z)
        self.az = lon.to(units.deg)
        self.alt = lat.to(units.deg)

        altaz = coordinates.AltAz(alt=alt_shift, az=az_shift)
        self.offset = coordinates.SkyOffsetFrame(origin=altaz)
        self.response = self.__offset_response

    def set_directions(self, alt, az):
        """

        Parameters
        ----------
        alt :

        az :


        Returns
        -------

        """
        self.__response.set_directions(alt, az)

    def set_location(self, location=None, lon=None, lat=None, height=None):
        """

        Parameters
        ----------
        location :
             (Default value = None)
        lon :
             (Default value = None)
        lat :
             (Default value = None)
        height :
             (Default value = None)

        Returns
        -------

        """

        if location is not None:
            self.location = location
        else:
            coords = dict(lon=lon, lat=lat, height=height)
            self.location = coordinates.EarthLocation(**coords)

    def noise(self):
        """ """

        band_width = numpy.diff(self.frequency_range)
        scaled_time = band_width * self.sampling_time
        noise_scale = numpy.sqrt(self.polarizations * scaled_time)
        minimum_temperature = self.system_temperature / noise_scale
        noise = minimum_temperature / self.__gain
        return self.noise_performance * noise.to(units.Jy)

    def __no_offset_response(self, altaz):
        """

        Parameters
        ----------
        altaz :


        Returns
        -------

        """

        return self.__response(altaz)

    def __offset_response(self, altaz):
        """

        Parameters
        ----------
        altaz :


        Returns
        -------

        """

        altazoff = altaz.transform_to(self.offset)
        altaz = coordinates.AltAz(alt=altazoff.lat, az=altazoff.lon)

        return self.__response(altaz)

    def __derived(self):
        """ """

        lon = self.__dict__.pop('lon')
        lat = self.__dict__.pop('lat')
        height = self.__dict__.pop('height')

        self.set_location(lon=lon, lat=lat, height=height)
        self.noise_performance = noise_performance[self.receiver_type]
        self.set_directivity(self.directivity)

    def set_directivity(self, directivity):
        """

        Parameters
        ----------
        directivity :


        Returns
        -------

        """

        self.directivity = numpy.atleast_1d(directivity)
        self.solid_angle = 4 * numpy.pi / directivity.to(1 / units.sr)
        reference_wavelength = (constants.c / self.reference_frequency)
        self.effective_area = reference_wavelength**2 / self.solid_angle.value
        self.effective_area = self.effective_area.to(units.meter**2)
        self.gain = 0.5 * (self.effective_area / constants.k_B)
        self.gain = self.gain.to(units.K / units.Jy)
        arg = 1 - self.solid_angle / (2 * numpy.pi * units.sr)
        radius = numpy.arccos(arg).to(units.deg)
        self.radius = numpy.atleast_1d(radius)

        self.beams = self.directivity.size
        if (self.beams == 1) and (self.az.size > 1):
            sizes = self.az.size, self.alt.size
            self.beams = numpy.unique(sizes)
            assert self.beams.size == 1
            self.beams = self.beams.item()

        value = numpy.full(self.beams, self.gain.value)
        self.__gain = value * self.gain.unit

        if self.kind in ('tophat', 'bessel', 'gaussian'):
            self.__response.set_radius(directivity)
