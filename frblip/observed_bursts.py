import os

import numpy

import pandas

from scipy.special import comb

from itertools import combinations

from astropy import coordinates, constants, units
from astropy.time import Time

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


def interferometry(*observations):

    """
    Perform a interferometry observation by a Radio Telescope array.
    """

    n_obs = len(observations)

    n_frb = int(numpy.unique([obs.n_frb for obs in observations]))
    n_channel = int(numpy.unique([obs.n_channel for obs in observations]))

    obsij = [
        cross_correlation(obsi, obsj)
        for obsi, obsj in combinations(observations, 2)
    ]

    n_comb = comb(n_obs, 2, exact=True)

    shapes = numpy.ones((n_comb, n_obs), dtype=numpy.int)

    idx_beam = numpy.row_stack([*combinations(range(n_obs), 2)])
    idx_comb = numpy.tile(numpy.arange(n_comb).reshape(-1, 1), (1, 2))

    n_beam = numpy.concatenate([obs.n_beam for obs in observations])
    shapes[idx_comb, idx_beam] = n_beam[idx_beam]

    _signal = [
        obs._signal.value.reshape((n_frb, *shape, 2 * n_channel + 1))
        for shape, obs in zip(shapes, obsij)
    ]

    _signal = sum(_signal, numpy.empty(()))

    _noise = [
        1 / obs._channels_noise.value.reshape((*shape, n_channel))**2
        for shape, obs in zip(shapes, obsij)
    ]

    _noise = sum(_noise, numpy.empty(()))
    _noise = numpy.sqrt(1 / _noise)

    return ObservedBursts(
        _signal, _noise, observations[0].frequency_bands.value,
        coordinates=observations[0].coordinates.obstime
    )


class ObservedBursts():

    def __init__(self, signal, noise, frequency_bands,
                 coordinates=None, location=None):

        self.location = location
        self.coordinates = coordinates

        self.frequency_bands = frequency_bands * units.MHz
        mid_frequency = 0.5 * (frequency_bands[1:] + frequency_bands[:-1])

        self.n_channel = len(frequency_bands) - 1

        self._band_widths = self.frequency_bands.diff()

        self._frequency = numpy.concatenate((frequency_bands, mid_frequency))
        self._frequency = numpy.sort(self._frequency) * units.MHz

        self._signal = signal * units.Jy
        self.n_frb = numpy.prod(signal.shape[0])
        self.n_beam = signal.shape[1:-1]

        self._channels_signal = simps(self._signal)
        self._channels_noise = noise * units.Jy

    def __getitem__(self, idx):

        return self.select(idx, inplace=False)

    def select(self, idx, inplace=False):

        _signal = self._signal[idx]
        coords = self.coordinates[idx]

        output = ObservedBursts(
            signal=_signal.value, noise=self._channels_noise.value,
            frequency_bands=self.frequency_bands.value,
            coordinates=coords, location=self.location
        )

        if not inplace:
            return output

        self._signal = _signal
        self.coordinates = coords
        self._channels_signal = self._channels_signal[idx]
        self.n_frb = self._signal.shape[0]

    def save(self, file):

        output = {
            'Signal (Jy)': self._signal.value,
            'Noise (Jy)': self._channels_noise.value,
            'Frequency Bands (MHz)': self.frequency_bands.value
        }

        if self.location is not None:

            output.update({
                'Longitude (degree)': self.location.lon.value,
                'Latitude (degree)': self.location.lat.value,
                'Elevation (meter)': self.location.height.value,
            })

        else:

            output['Location'] = None

        if type(self.coordinates) is Time:

            output.update({
                'Coordinates': 'Time',
                'Time': self.coordinates
            })

        elif type(self.coordinates) is coordinates.SkyCoord:

            output.update({
                'Coordinates': 'Sky',
                'Time': self.coordinates.obstime,
                'Altitude (deg)': self.coordinates.alt.value,
                'Azimuth (deg)': self.coordinates.az.value,
            })

        numpy.savez(file, **output)

    @staticmethod
    def load(name):

        file = numpy.load(name, allow_pickle=True)

        if 'Location' not in file.files:

            location = coordinates.EarthLocation(
                lon=file['Longitude (degree)'] * units.degree,
                lat=file['Latitude (degree)'] * units.degree,
                height=file['Elevation (meter)'] * units.meter
            )

        else:

            location = None

        if file['Coordinates'] == 'Time':

            coords = file['Time']

        elif file['Coordinates'] == 'Sky':

            coords = coordinates.AltAz(
                az=file['Azimuth (deg)'] * units.degree,
                alt=file['Altitude (deg)'] * units.degree,
                obstime=file['Time']
            )

        return ObservedBursts(
            signal=file['Signal (Jy)'], noise=file['Noise (Jy)'],
            frequency_bands=file['Frequency Bands (MHz)'],
            coordinates=coords, location=location
        )

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
