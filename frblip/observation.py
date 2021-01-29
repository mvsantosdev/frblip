import os

import numpy

import pandas

from scipy.special import comb

from itertools import combinations

from astropy import coordinates, constants, units
from astropy.time import Time

from .utils import simps


class Observation():

    def __init__(self, response, noise, frequency_bands,
                 sampling_time=0, coordinates=None,
                 full=True):

        self.full = full

        self.sampling_time = sampling_time
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

        self.noise = noise

    def split_beams(self):

        n_beam = numpy.prod(self.n_beam)
        shape = n_beam, self.n_channel

        noises = numpy.split(self._channels_noise, n_beam, 0)

        responses = numpy.split(self.response, n_beam, -1)

        return [
            Observation(response, noise,
                        self.frequency_bands,
                        self.coordinates)
            for response, noise in zip(responses, noises)
        ]

    def __getitem__(self, idx):

        return self.select(idx, inplace=False)

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
