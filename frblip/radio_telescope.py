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
from .utils import load_params, sub_dict


noise_performance = {
    'total-power': 1,
    'switched': 2,
    'correlation': numpy.sqrt(2),
    '1bit-digital': 2.21,
    '2bit-digital': 1.58
}


class RadioTelescope(object):

    """
    Class which defines a Radio Surveynp
    """

    def __init__(self, name='bingo', kind='gaussian', start_time=None,
                 offset=(90.0, 0.0), **kwargs):

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

        self.start_time = Time.now() if start_time is None else start_time

        name_ = '{}/{}.npz'.format(_DATA, name)

        name_ = name_ if os.path.exists(name_) else name

        input_dict = load_params(name_)

        input_dict['kind'] = kind

        keys = ['grid', 'xrange', 'yrange']
        grid_params = sub_dict(input_dict, keys=keys, pop=True)

        if kind == 'grid':

            coords = sub_dict(input_dict, keys=['alt', 'az'])
            grid_params.update(coords)

            input_dict['_response'] = CartesianGrid(**grid_params)

        elif kind in ('tophat', 'bessel', 'gaussian'):

            keys = ['kind', 'directivity', 'alt', 'az']
            pattern_params = sub_dict(input_dict, keys=keys)

            input_dict['_response'] = FunctionalPattern(**pattern_params)

        else:

            print('Please choose a valid pattern kind')

        self.__dict__.update(input_dict)
        self._derived()

        self._set_offset(offset)

    def _set_offset(self, offset):

        alt, az = offset
        alt = alt - 90

        altaz = coordinates.AltAz(alt=alt * units.degree,
                                  az=az * units.degree)

        self.offset = coordinates.SkyOffsetFrame(origin=altaz)

    def noise(self):

        band_widths = numpy.diff(self.frequency_bands)

        scaled_time = band_widths * self.sampling_time
        noise_scale = numpy.sqrt(self.polarizations * scaled_time)

        minimum_temperature = self.system_temperature / noise_scale

        noise = minimum_temperature / self.gain.reshape(-1, 1)

        return self.noise_performance * noise.to(units.Jy)

    def response(self, altaz):

        altazoff = altaz.transform_to(self.offset)

        altaz = coordinates.AltAz(alt=altazoff.lat,
                                  az=altazoff.lon)

        return self._response(altaz)

    def _derived(self):

        lon = self.__dict__.pop('lon')
        lat = self.__dict__.pop('lat')
        height = self.__dict__.pop('height')

        self.location = coordinates.EarthLocation(lon=lon, lat=lat,
                                                  height=height)

        self.noise_performance = noise_performance[self.receiver_type]

        self.solid_angle = 4 * numpy.pi / self.directivity.to(1 / units.sr)

        reference_wavelength = (constants.c / self.reference_frequency)

        self.effective_area = reference_wavelength**2 / self.solid_angle.value
        self.effective_area = self.effective_area.to(units.meter**2)

        self.gain = 0.5 * (self.effective_area / constants.k_B)
        self.gain = self.gain.to(units.K / units.Jy)
