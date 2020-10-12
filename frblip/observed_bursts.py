import os

import numpy

import pandas

from astropy import coordinates, units, constants

from .utils import simps


def cross_correlation(obsi, obsj):

    """
    Compute the cross correlations between two observation sets
    """

    xi, yi, zi = obsi.location.geocentric
    xj, yj, zj = obsj.location.geocentric

    distance = numpy.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)

    alti = obsi.coordinates.alt
    altj = obsj.coordinates.alt

    arc = .5 * (alti + alti)
    optical_path = distance * numpy.cos(arc).reshape(-1, 1)

    freq = obsi._frequency

    arg = (2 * numpy.pi * freq * optical_path / constants.c).decompose().value
    cos = numpy.cos(arg[:, numpy.newaxis, numpy.newaxis, :])

    signal = obsi._signal[:, :, numpy.newaxis] * obsj._signal[:, numpy.newaxis]
    signal = 0.5 * numpy.sqrt(signal) * cos

    noii = obsi.noise(channels=True)
    noij = obsj.noise(channels=True)

    noise = numpy.sqrt(0.5 * noii[:, numpy.newaxis] * noij)

    return ObservedBursts(signal, noise, obsi.frequency_bands,
                          coordinates=obsi.coordinates.obstime)


class ObservedBursts():

    def __init__(self, signal, noise, frequency_bands,
                 coordinates=None, location=None):

        self.location = location
        self.coordinates = coordinates

        self.frequency_bands = frequency_bands * units.MHz
        mid_frequency = 0.5 * (frequency_bands[1:] + frequency_bands[:-1])

        self._band_widths = self.frequency_bands.diff()

        self._frequency = numpy.concatenate((frequency_bands, mid_frequency))
        self._frequency = numpy.sort(self._frequency) * units.MHz
        self._signal = signal * units.Jy

        self._channels_signal = simps(self._signal)
        self._channels_noise = noise * units.Jy

    def __getitem__(self, idx):

        return self.select(idx, inplace=False)

    def select(self, idx, inplace=False):

        frbs = self if inplace else ObservedBursts()

        frbs.coordinates = self.coordinates[idx]

        frbs._signal = self._signal[idx]
        frbs._channels_signal = self._channels_signal[idx]

        if not inplace:

            frbs._channels_noise = self.noise
            frbs.location = self.location

            frbs.frequency_bands = self.frequency_bands
            frbs._band_widths = self._band_widths

            frbs._frequency = self._frequency

            return frbs

    def save(self, file):

        output = {
            'Time': self.time,
            'Signal (K)': self.signal,
            'Noise (K)': self.noise,
            'Longitude (degree)': self.location.lon,
            'Latitude (degree)': self.location.lat,
            'Elevation (meter)': self.location.height
        }

        numpy.savez(file, **output)

    def signal(self, channels=False):

        if channels:

            return self._channels_signal

        return numpy.average(self._channels_signal,
                             weights=self._band_widths,
                             axis=-1)

    def noise(self, channels=False):

        if channels:

            return self._channels_noise

        inoise = (1 / self._channels_noise**2).sum(-1)

        return 1 / numpy.sqrt(inoise)

    def signal_to_noise(self, channels=False):

        signal = self.signal(channels)
        noise = self.noise(channels)

        return (signal / noise).value
