import os

import numpy

import pandas

from astropy import coordinates, units


class ObservedBursts():

    def __init__(self, file=None):

        if file is not None:

            try:

                data = numpy.load(file, allow_pickle=True)

                self.signal = data['Signal (K)'] * units.K
                self.noise = data['Noise (K)'] * units.K

                self.time = data['Time']

                self.location = coordinates.EarthLocation(
                    lon=data['Longitude (degree)'],
                    lat=data['Latitude (degree)'],
                    height=data['Elevation (meter)']
                )

            except FileNotFoundError:

                print('Please provide a valid file.')

    def __getitem__(self, idx):
    
        return self.select(idx, inplace=False)

    def select(self, idx, inplace=False):

        frbs = self if inplace else ObservedBursts()

        frbs.signal = self.signal[:, idx, :]
        frbs.time = self.time[idx]

        if not inplace:

            frbs.noise = self.noise
            frbs.location = self.location

            return frbs

    def save(self, file):

        output = {
            'Signal (K)': self.signal,
            'Noise (K)': self.noise,
            'Time': self.time.to_datetime(),
            'Longitude (degree)': self.location.lon,
            'Latitude (degree)': self.location.lat,
            'Elevation (meter)': self.location.height
        }

        numpy.savez(file, **output)

    def signal_to_noise(self, beams=None, bands=None):

        signal_sq = self.signal**2
        noise_sq = self.noise**2

        SNR_sq = (signal_sq / noise_sq).sum(0).sum(-1)

        return numpy.sqrt(SNR_sq)
