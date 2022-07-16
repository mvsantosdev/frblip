import os
import sys
import warnings

import bz2
import dill

import numpy
import xarray
from sparse import COO

from functools import partial

from astropy import units, constants, coordinates
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator

from .observation import Observation, Interferometry


class BasicSampler(object):

    def __len__(self):
        return self.size

    def __getattr__(self, attr, *args, **kwargs):

        name = '_{}'.format(attr)

        if name in dir(self):
            def func(*names, **kwargs):
                f = partial(getattr(self, name), **kwargs)
                keys = self.observations.keys() if names == () else names
                if len(keys) == 1:
                    return f(*keys)
                return {
                    key: f(key) for key in keys
                }
            return func
        else:
            class_name = type(self).__name__
            error = "'{}' object has no attribute '{}'"
            raise AttributeError(error.format(class_name, attr))

    def obstime(self, location):

        loc = location.get_itrs()
        loc = loc.cartesian.xyz

        path = loc @ self.xyz
        time_delay = path / constants.c

        return self.itrs_time - time_delay

    def altaz_from_location(self, location, interp=300):

        lon = location.lon
        lat = location.lat
        height = location.height

        print(
            'Computing positions for {} sources'.format(self.size),
            'at site lon={:.3f}, lat={:.3f},'.format(lon, lat),
            'height={:.3f}.'.format(height), end='\n\n'
        )

        obstime = self.obstime(location)
        frame = coordinates.AltAz(location=location, obstime=obstime)
        interp_time = interp * units.s

        with erfa_astrom.set(ErfaAstromInterpolator(interp_time)):
            return self.icrs.transform_to(frame)

    def __observe(self, telescope, name=None, sparse=True, dtype=numpy.double):

        print('Performing observation for telescope {}...'.format(name))

        if 'observations' not in self.__dict__:
            self.observations = {}

        n_obs = len(self.observations)
        obs_name = 'OBS_{}'.format(n_obs) if name is None else name
        obs_name = obs_name.replace(' ', '_')

        sampling_time = telescope.sampling_time
        frequency_range = telescope.frequency_range

        if 'altaz' in dir(self):
            altaz = self.altaz
        else:
            location = telescope.location
            altaz = self.altaz_from_location(location)

        mask = altaz.alt > 0

        dims = self.kind, obs_name

        resp = telescope.response(altaz[mask])
        shape = self.size, *resp.shape[1:]
        response = numpy.zeros(shape, dtype=dtype)
        response[mask] = resp
        if sparse:
            response = COO(response)
        response = xarray.DataArray(response, dims=dims, name='Response')

        noi = telescope.noise
        noise = xarray.DataArray(noi.value, dims=obs_name, name='Noise')
        noise.attrs['unit'] = noi.unit

        time_delay = telescope.time_array(altaz)
        if time_delay is not None:
            unit = time_delay.unit
            time_delay = xarray.DataArray(time_delay.value, dims=dims,
                                          name='Time Delay')
            time_delay.attrs['unit'] = unit

        if 'altaz' in dir(self):
            altaz = None

        observation = Observation(response, noise, time_delay, frequency_range,
                                  sampling_time, altaz)

        self.observations[obs_name] = observation

    def observe(self, telescopes, name=None, location=None, sparse=False,
                dtype=numpy.double, verbose=True):

        old_target = sys.stdout
        sys.stdout = old_target if verbose else open(os.devnull, 'w')

        if 'altaz' not in dir(self):
            if isinstance(location, coordinates.EarthLocation):
                loc = location
            elif isinstance(location, str):
                if location in telescopes:
                    loc = telescopes[location].location
                else:
                    loc = coordinates.EarthLocation.of_site(location)

                self.altaz = self.altaz_from_location(loc)
            elif location is not None:
                error = '{} is not a valid location'.format(location)
                raise TypeError(error)

        if type(telescopes) is dict:
            for name, telescope in telescopes.items():
                self.__observe(telescope, name, sparse, dtype)
        else:
            self.__observe(telescopes, name, sparse, dtype)

        sys.stdout = old_target

    def interferometry(self, namei, namej=None, reference=False,
                       degradation=None, overwrite=False):

        if reference:
            names = [
                name for name in self.observations
                if (name != namei) and ('INTF' not in name)
            ]
            for namej in names:
                self.interferometry(namej)
                self.interferometry(namej, namei)
        else:
            obsi, obsj = self[namei], self[namej]
            if namej is None:
                key = 'INTF_{}'.format(namei)
            else:
                key = 'INTF_{}_{}'.format(namei, namej)

            if (key not in self.observations) or overwrite:
                interferometry = Interferometry(obsi, obsj, degradation)
                self.observations[key] = interferometry
            else:
                warning_message = '{} is already computed. '.format(key) + \
                                  'You may set overwrite=True to recompute.'
                warnings.warn(warning_message)

    def iterfrbs(self, start=0, stop=None, step=1):

        stop = self.size if stop is None else stop
        for i in range(start, stop, step):
            yield self[i]

    def iterchunks(self, size=1, start=0, stop=None, retindex=False):

        stop = self.size if stop is None else stop

        if retindex:
            for i in range(start, stop, size):
                j = i + size
                yield i, j, self[i:j]
        else:
            for i in range(start, stop, size):
                j = i + size
                yield self[i:j]

    def clean(self, names=None):

        if hasattr(self, 'observations'):
            if names is None:
                del self.observations
            elif isinstance(names, str):
                del self.observations[names]
            else:
                for name in names:
                    del self.observations[name]

    def copy(self, clear=False):

        copy = dill.copy(self)
        keys = self.__dict__.keys()

        if clear:
            class_name = type(self).__name__
            for key in keys:
                if '_{}__'.format(class_name) in key:
                    delattr(copy, key)
        return copy

    def save(self, name):

        file_name = '{}.blips'.format(name)
        file = bz2.BZ2File(file_name, 'wb')
        copy = self.copy()
        dill.dump(copy, file, dill.HIGHEST_PROTOCOL)
        file.close()

    @staticmethod
    def load(file):

        file_name = '{}.blips'.format(file)
        file = bz2.BZ2File(file_name, 'rb')
        loaded = dill.load(file)
        file.close()
        return loaded