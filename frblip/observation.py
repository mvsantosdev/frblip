import os

import numpy

import pandas

from scipy.special import comb

from itertools import combinations

from astropy import coordinates, constants, units
from astropy.time import Time

from .utils import simps, null_coordinates, null_location, null_obstime


class Observation():

    def __init__(self, response, noise, frequency_bands,
                 sampling_time=0, coordinates=None,
                 full=True):

        self.full = full
        self.sampling_time = sampling_time

        if coordinates is None:

            obstime = null_obstime(iso=numpy.nan)
            location = null_location(lon=numpy.nan, lat=numpy.nan,
                                     height=numpy.nan)

            self.coordinates = null_coordinates(az=numpy.nan,
                                                alt=numpy.nan,
                                                obstime=obstime,
                                                location=location)

        else:

            self.coordinates = coordinates

        self.response = response

        self.frequency_bands = frequency_bands
        mid_frequency = 0.5 * (frequency_bands[1:] + frequency_bands[:-1])
        self.n_channel = len(frequency_bands) - 1
        self._band_widths = self.frequency_bands.diff()
        self._frequency = numpy.concatenate((frequency_bands, mid_frequency))
        self._frequency = numpy.sort(self._frequency)

        self.n_beam = response.shape[1:]
        self.n_telescopes = len(self.n_beam)

        if noise.shape == (*self.n_beam, self.n_channel):

            self.noise = noise

        elif noise.shape == (1, self.n_channel):

            self.noise = numpy.tile(noise, (*self.n_beam, 1))
            self.unique_beam = True

    def split_beams(self):

        n_beam = numpy.prod(self.n_beam)
        shape = n_beam, self.n_channel

        responses = numpy.split(self.response, n_beam, -1)

        if self.noise.shape == (*self.n_beam, self.n_channel):

            noises = numpy.split(self.noise, n_beam, 0)

            return [
                Observation(response, noise,
                            self.frequency_bands,
                            self.sampling_time,
                            self.coordinates,
                            self.full)
                for response, noise in zip(responses, noises)
            ]

        elif self.noise.shape == (1, self.n_channel):

            return [
                Observation(response, self.noise,
                            self.frequency_bands,
                            self.sampling_time,
                            self.coordinates,
                            self.full)
                for response in responses
            ]

    def __getitem__(self, idx):

        return self.select(idx, inplace=False)

    def to_dict(self, key='OBS'):

        out_dict = {
            '{}__response'.format(key): self.response,
            '{}__sampling_time'.format(key): self.sampling_time,
            '{}__frequency_bands'.format(key): self.frequency_bands,
            '{}__full'.format(key): self.full
        }

        if self.__dict__.get('unique_beam'):
            out_dict['{}__noise'.format(key)] = self.noise[0]
        else:
            out_dict['{}__noise'.format(key)] = self.noise

        if self.coordinates is not None:

            out_dict.update({
                '{}__az'.format(key): self.coordinates.az,
                '{}__alt'.format(key): self.coordinates.alt,
                '{}__obstime'.format(key): self.coordinates.obstime.iso,
                '{}__lon'.format(key): self.coordinates.location.lon,
                '{}__lat'.format(key): self.coordinates.location.lat,
                '{}__height'.format(key): self.coordinates.location.height
            })

        return out_dict

    @staticmethod
    def from_dict(key='', **kwargs):

        response = kwargs.get('{}__response'.format(key))
        noise = kwargs.get('{}__noise'.format(key))
        sampling_time = kwargs.get('{}__sampling_time'.format(key))
        frequency_bands = kwargs.get('{}__frequency_bands'.format(key))
        full = kwargs.get('{}__full'.format(key))

        frequency_bands = frequency_bands * units.MHz
        sampling_time = sampling_time * units.ms
        noise = noise * units.Jy

        lon = kwargs.get('{}__lon'.format(key))
        lat = kwargs.get('{}__lat'.format(key))
        height = kwargs.get('{}__height'.format(key))

        az = kwargs.get('{}__az'.format(key))
        alt = kwargs.get('{}__alt'.format(key))
        obstime = kwargs.get('{}__obstime'.format(key))

        if numpy.isfinite([lon, lat, height]).all():

            lon = lon * units.degree
            lat = lat * units.degree
            height = height * units.meter

            location = coordinates.EarthLocation(lon=lon, lat=lat,
                                                 height=height)

            az_finite = numpy.isfinite(az)
            alt_finite = numpy.isfinite(alt)

            if numpy.logical_and(az_finite, alt_finite).all():

                az = az * units.degree
                alt = alt * units.degree
                obstime = Time(obstime)

                local_coordinates = coordinates.AltAz(az=az, alt=alt,
                                                      obstime=obstime,
                                                      location=location)

                return Observation(response, noise, frequency_bands,
                                   sampling_time, local_coordinates,
                                   full)

            obstime = null_obstime(iso=obstime)
            local_coordinates = null_coordinates(az=az, alt=alt,
                                                 obstime=obstime,
                                                 location=location)

            return Observation(response, noise, frequency_bands,
                               sampling_time, local_coordinates,
                               full)

        obstime = null_obstime(iso=obstime)
        location = null_location(lon=lon, lat=lat, height=height)

        local_coordinates = null_coordinates(az=az, alt=alt,
                                             obstime=obstime,
                                             location=location)

        return Observation(response, noise, frequency_bands,
                           sampling_time, local_coordinates, full)

    def select(self, idx, inplace=False):

        idx = idx.ravel() if idx.ndim == 2 else idx

        coords = self.coordinates[idx] if self.full else self.coordinates
        response = self.response[idx] if self.full else self.response

        noise = self.noise
        sampling_time = self.sampling_time
        frequency_bands = self.frequency_bands

        if not inplace:

            output = Observation(response, noise, frequency_bands,
                                 sampling_time, coords)

            return output

        self.coordinates = coords
        self.response = response
