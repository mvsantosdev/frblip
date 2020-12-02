import os

import numpy

import pandas

from scipy.special import comb

from itertools import combinations

from astropy import coordinates, constants, units
from astropy.time import Time

from .utils import simps


def _cross_correlation(obsi, obsj, interference=False):

    """
    Compute the cross correlations between two observation sets
    """

    arg = 0.0

    if interference:

        xi, yi, zi = obsi.location.geocentric
        xj, yj, zj = obsj.location.geocentric

        distance = numpy.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)

        alti = obsi.coordinates.alt
        altj = obsj.coordinates.alt

        arc = .5 * (alti + altj)
        optical_path = distance * numpy.cos(arc).reshape(-1, 1)

        freq = obsi._frequency

        arg = 2 * numpy.pi * freq * optical_path / constants.c
        arg = arg.decompose().value

    respi = obsi.response
    respj = obsj.response

    response = numpy.sqrt(respi[..., numpy.newaxis] * respj[:, numpy.newaxis])
    signal = numpy.sqrt(obsi._signal * obsj._signal) * numpy.cos(arg)
    time_factor = numpy.sqrt(obsi.time_factor * obsj.time_factor)

    noii = obsi.noise(channels=True)
    noij = obsj.noise(channels=True)

    noise = numpy.sqrt(0.5 * noii[:, numpy.newaxis] * noij)

    return ObservedBursts(signal.value, response,
                          time_factor, noise.value,
                          obsi.frequency_bands.value)


def interferometry(*observations, interference=False):

    """
    Perform a interferometry observation by a Radio Telescope array.
    """

    n_obs = len(observations)

    if n_obs == 1:

        return interferometry(*observations[0].split_beams(),
                              interference=interference)

    if n_obs == 2:

        out = _cross_correlation(*observations, interference=interference)
        out.response = numpy.squeeze(out.response)
        out._channels_noise = numpy.squeeze(out._channels_noise)

        return out

    n_frb = int(numpy.unique([obs.n_frb for obs in observations]))
    n_channel = int(numpy.unique([obs.n_channel for obs in observations]))

    obsij = [
        _cross_correlation(obsi, obsj, interference=interference)
        for obsi, obsj in combinations(observations, 2)
    ]

    n_comb = comb(n_obs, 2, exact=True)

    shapes = numpy.ones((n_comb, n_obs), dtype=numpy.int)

    idx_beam = numpy.row_stack([*combinations(range(n_obs), 2)])
    idx_comb = numpy.tile(numpy.arange(n_comb).reshape(-1, 1), (1, 2))

    n_beam = numpy.concatenate([obs.n_beam for obs in observations])
    shapes[idx_comb, idx_beam] = n_beam[idx_beam]

    signal = numpy.stack([obs._signal for obs in obsij]).mean(0)

    response = [
        obs.response.reshape((n_frb, *shape))
        for shape, obs in zip(shapes, obsij)
    ]

    ishape = numpy.ones(n_obs, dtype=int)

    time_factor = [
        obs.time_factor.reshape((n_frb, *ishape))
        for shape, obs in zip(shapes, obsij)
    ]

    response = [t * r for t, r in zip(time_factor, response)]
    response = sum(response, numpy.empty(()))

    noise = [
        1 / obs._channels_noise.value.reshape((*shape, n_channel))**2
        for shape, obs in zip(shapes, obsij)
    ]

    noise = sum(noise, numpy.empty(()))
    noise = numpy.sqrt(1 / noise)

    out = ObservedBursts(
        signal=signal.value,
        response=response,
        time_factor=numpy.array(1.0),
        noise=noise,
        frequency_bands=observations[0].frequency_bands.value,
        coordinates=observations[0].coordinates.obstime
    )

    out.response = numpy.squeeze(out.response)
    out._channels_noise = numpy.squeeze(out._channels_noise)

    return out


class ObservedBursts():

    def __init__(self, signal, response, time_factor,
                 noise, frequency_bands, coordinates=None,
                 location=None):

        self.location = location
        self.coordinates = coordinates

        self.response = response
        self.time_factor = time_factor

        self.frequency_bands = frequency_bands * units.MHz
        mid_frequency = 0.5 * (frequency_bands[1:] + frequency_bands[:-1])

        self.n_channel = len(frequency_bands) - 1

        self._band_widths = self.frequency_bands.diff()

        self._frequency = numpy.concatenate((frequency_bands, mid_frequency))
        self._frequency = numpy.sort(self._frequency) * units.MHz

        self._signal = signal * units.Jy
        self.n_frb = numpy.prod(signal.shape[0])
        self.n_beam = response.shape[1:]
        self.n_telescopes = len(self.n_beam)

        self._channels_signal = simps(self._signal)
        self._channels_noise = noise * units.Jy

    def __len__(self):

        return self.n_frb

    def __getitem__(self, idx):

        return self.select(idx, inplace=False)

    def select(self, idx, inplace=False):

        _signal = self._signal[idx]
        coords = self.coordinates[idx]
        response = self.response[idx]
        time_factor = self.time_factor[idx]

        output = ObservedBursts(
            signal=_signal.value,
            response=response,
            time_factor=time_factor,
            noise=self._channels_noise.value,
            frequency_bands=self.frequency_bands.value,
            coordinates=coords, location=self.location
        )

        if not inplace:
            return output

        self._signal = _signal
        self.coordinates = coords
        self._channels_signal = self._channels_signal[idx]
        self.n_frb = self._signal.shape[0]

    def save(self, file, compressed=True):

        output = {
            'Signal (Jy)': self._signal.value,
            'Noise (Jy)': self._channels_noise.value,
            'Time Factor': self.time_factor,
            'Beam Response': self.response,
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

        if compressed:

            numpy.savez_compressed(file, **output)
        else:

            numpy.savez(file, **output)

    def split_beams(self):

        n_beam = numpy.prod(self.n_beam)

        shape = n_beam, self.n_channel
        _channels_noise = self._channels_noise.reshape(shape)
        _channels_noise = numpy.split(_channels_noise, n_beam, 0)

        response = self.response.reshape((self.n_frb, n_beam))
        response = numpy.split(response, n_beam, -1)

        return [
            ObservedBursts(self._signal.value, r,
                           self.time_factor, n.value,
                           self.frequency_bands.value,
                           self.coordinates,
                           self.location)
            for r, n in zip(response, _channels_noise)
        ]

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
            signal=file['Signal (Jy)'], response=file['Beam Response'],
            time_factor=file['Time Factor'], noise=file['Noise (Jy)'],
            frequency_bands=file['Frequency Bands (MHz)'],
            coordinates=coords, location=location
        )

    def signal(self, channels=False):

        nshape = len(self.response.shape) - 1
        ishape = numpy.ones(nshape, dtype=int)

        if channels:

            signal = self._channels_signal
            signal = signal * self.time_factor.reshape(-1, 1)
            signal = signal.reshape(self.n_frb, *ishape, self.n_channel)

            return self.response[..., numpy.newaxis] * signal

        signal = numpy.average(self._channels_signal,
                               weights=self._band_widths,
                               axis=-1)
        signal = self.time_factor * signal
        signal = signal.reshape(self.n_frb, *ishape)

        return self.response * signal

    def noise(self, channels=False):

        if channels:

            return self._channels_noise

        inoise = (1 / self._channels_noise**2).sum(-1)

        return 1 / numpy.sqrt(inoise)

    def signal_to_noise(self, channels=False):

        signal = self.signal(channels)
        noise = self.noise(channels)

        return (signal / noise).value
