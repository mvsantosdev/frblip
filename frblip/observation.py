import os

import numpy

import pandas

from scipy.special import comb

from itertools import combinations

from astropy.time import Time
from astropy import coordinates, constants, units

from .utils import sub_dict


class Observation():

    def __init__(self, response=None, noise=None, frequency_bands=None,
                 sampling_time=None, altaz=None):

        self.sampling_time = sampling_time

        self.response = response
        self.noise = noise
        self.frequency_bands = frequency_bands
        self.altaz = altaz

        self.__set_frequencies()
        self.__set_response()

    def __set_response(self):

        noise = self.noise is not None
        response = self.response is not None

        if response and noise:

            self.__n_beam = self.response.shape[1:]
            self.__n_telescopes = numpy.size(self.__n_beam)

            if self.noise.shape == (1, self.__n_channel):
                self.noise = numpy.tile(self.noise, (*self.__n_beam, 1))

    def __set_frequencies(self):

        frequency_bands = getattr(self, 'frequency_bands')
        if frequency_bands:
            self.__n_channel = frequency_bands.size - 1
            self.__band_widths = frequency_bands.diff()

    def __set_coordinates(self):

        params = self.__dict__

        location = sub_dict(params, pop=True, keys=['lon', 'lat', 'height'])
        kwargs = sub_dict(params, pop=True, keys=['alt', 'az', 'obstime'])
        kwargs['obstime'] = Time(kwargs['obstime'], format='mjd')

        if location:
            kwargs['location'] = coordinates.EarthLocation(**location)

        self.altaz = coordinates.AltAz(**kwargs)

    def split_beams(self):

        n_beam = numpy.prod(self.n_beam)
        shape = n_beam, self.n_channel
        responses = numpy.split(self.response, n_beam, -1)
        noises = numpy.split(self.noise, n_beam, 0)

        return [
            Observation(response, noise, self.frequency_bands,
                        self.sampling_time, self.altaz)
            for response, noise in zip(responses, noises)
        ]

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
            obstime = getattr(altaz, 'obstime', None)
            out_dict['obstime'] = obstime.mjd

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

        output = Observation.__new__(Observation)
        output.__dict__.update(kwargs)
        output.__set_coordinates()
        output.__set_frequencies()
        output.__set_response()

        return output

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            return self.select(idx)
        idx = numpy.array(idx)
        return self.select(idx)

    def select(self, idx, inplace=False):
        response = self.response[idx]
        altaz = getattr(self, 'altaz', None)
        altaz = altaz[idx] if altaz else None
        if not inplace:
            output = Observation.__new__(Observation)
            output.__dict__.update(self.__dict__)
            output.response = response
            return output
        self.response = response
