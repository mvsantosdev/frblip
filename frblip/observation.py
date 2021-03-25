import os

import numpy

import pandas

from scipy.special import comb

from itertools import combinations

from astropy import coordinates, constants, units
from astropy.time import Time

from .utils import sub_dict


class Observation():

    def __init__(self, response=None, noise=None, frequency_bands=None,
                 sampling_time=None, altaz=None):

        self.sampling_time = sampling_time

        self.response = response
        self.noise = noise
        self.frequency_bands = frequency_bands
        self.altaz = altaz

        self._set_frequencies()
        self._set_response()

    def _set_response(self):

        response = self.response is not None
        noise = self.noise is not None

        if response and noise:

            self.__n_beam = self.response.shape[1:]
            self.__n_telescopes = numpy.size(self.__n_beam)

            if self.noise.shape == (1, self.__n_channel):
                self.noise = numpy.tile(self.noise, (*self.__n_beam, 1))

    def _set_frequencies(self):

        if self.frequency_bands is not None:

            mid = (self.frequency_bands[1:] + self.frequency_bands[:-1]) / 2
            self.__n_channel = len(self.frequency_bands) - 1
            self.__band_widths = self.frequency_bands.diff()
            self.__frequency = numpy.concatenate((self.frequency_bands, mid))
            self.__frequency = numpy.sort(self.__frequency)

    def _set_coordinates(self):

        params = self.__dict__

        kwargs = sub_dict(params, pop=True, keys=['alt', 'az', 'obstime'])
        location = sub_dict(params, pop=True, keys=['lon', 'lat', 'height'])

        if location:
            kwargs['location'] = coordinates.EarthLocation(**location)

        self.altaz = coordinates.AltAz(**kwargs)

    def split_beams(self):

        n_beam = numpy.prod(self.n_beam)
        shape = n_beam, self.n_channel

        responses = numpy.split(self.response, n_beam, -1)
        noises = numpy.split(self.noise, n_beam, 0)

        return [
            Observation(response, noise,
                        self.frequency_bands,
                        self.sampling_time,
                        self.altaz)
            for response, noise in zip(responses, noises)
        ]

    def __getitem__(self, idx):

        return self.select(idx, inplace=False)

    def to_dict(self, flag=''):

        out_dict = {
            key: value
            for key, value in self.__dict__.items()
            if '_Observation__' not in key
            and value is not None
        }

        altaz = out_dict.pop('altaz', None)

        if altaz:

            out_dict['az'] = getattr(altaz, 'az', None)
            out_dict['alt'] = getattr(altaz, 'alt', None)
            out_dict['obstime'] = getattr(altaz, 'obstime', None)

            location = getattr(altaz, 'location', None)

            if location:

                out_dict['lon'] = getattr(location, 'lon', None)
                out_dict['lat'] = getattr(location, 'lat', None)
                out_dict['height'] = getattr(location, 'height', None)

        out_dict = {
            '{}{}'.format(flag, key): value
            for key, value in out_dict.items()
            if value is not None
        }

        return out_dict

    @staticmethod
    def from_dict(kwargs):

        output = Observation()

        output.__dict__.update(kwargs)

        output._set_coordinates()
        output._set_frequencies()
        output._set_response()

        return output

    def select(self, idx, inplace=False):

        idx = idx.ravel() if idx.ndim == 2 else idx

        altaz = self.altaz[idx] if len(self.altaz) > 0 else self.altaz
        response = self.response[idx]

        noise = self.noise
        sampling_time = self.sampling_time
        frequency_bands = self.frequency_bands

        if not inplace:

            output = Observation(response, noise, frequency_bands,
                                 sampling_time, altaz)

            return output

        self.altaz = altaz
        self.response = response
