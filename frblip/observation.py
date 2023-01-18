import dill

import numpy
import xarray

from itertools import combinations

from scipy.special import erf
from astropy import coordinates, units


def scattered_gaussian(t, w, ts, t0=0.0):
    """

    Parameters
    ----------
    t :

    w :

    ts :

    t0 :
         (Default value = 0.0)

    Returns
    -------

    """

    x = .5 * (w / ts)**2
    f = numpy.exp(x)

    x = (t0 - t) / ts
    g = numpy.exp(x)

    x = t - t0 - w**2 / ts
    y = w * numpy.sqrt(2)
    h = 1 + erf(x / y)

    ff = f * g * h

    return ff / ff.max(0)


class Observation():
    """ """

    def __init__(self, response=None, noise=None, time_delay=None,
                 frequency_range=None, sampling_time=None, altaz=None):

        self.response = response
        self.noise = noise
        if time_delay is not None:
            self.time_delay = time_delay
        self.frequency_range = frequency_range
        self.sampling_time = sampling_time
        if altaz is not None:
            self.altaz = altaz

        self.kind = response.dims[0]
        self.width = self.frequency_range.diff()

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            return self.select(idx)
        idx = numpy.array(idx)
        return self.select(idx)

    def __set_coordinates(self):

        params = self.__dict__
        location = {
            key: params.pop(key)
            for key in ('lon', 'lat', 'height')
        }
        kwargs = {
            key: params.pop(key)
            for key in ('alt', 'az', 'obstime')
        }

        if location:
            kwargs['location'] = coordinates.EarthLocation(**location)

        self.altaz = coordinates.AltAz(**kwargs)

    def get_obstime(self):
        """ """

        if hasattr(self, 'altaz'):
            return self.altaz.obstime
        size = self.response.sizes[self.kind]
        return numpy.zeros(size) * units.ms

    def get_time_delay(self):
        """ """

        if hasattr(self, 'time_delay'):
            return self.time_delay
        values = numpy.zeros_like(self.response.as_numpy())
        dims = self.response.dims
        return xarray.DataArray(values, dims=dims,
                                attrs={'unit': units.ms})

    def get_response(self, total=False):

        response = self.response

        if total:
            if isinstance(total, str) and (total in response.dims):
                levels = total
            if isinstance(total, list):
                levels = [*filter(lambda x: x in response.dims, total)]
            elif total is True:
                levels = [*filter(lambda x: x not in (self.kind, 'CHANNEL'),
                                  response.dims)]
            response = (response**2).sum(dim=levels)
            response = numpy.sqrt(response)

        return response

    def get_noise(self, total=False, channels=1, minimum=False):
        """

        Parameters
        ----------
        total :
             (Default value = False)
        channels :
             (Default value = 1)
        minimum : Bool
             (Default value = False)

        Returns
        -------

        """

        noise = numpy.full(channels, numpy.sqrt(channels))
        noise = xarray.DataArray(noise, dims='CHANNEL')
        noise = self.noise * noise
        if not minimum:
            noise = (1 / self.response) * noise

        if total:
            if isinstance(total, str) and (total in noise.dims):
                levels = total
            if isinstance(total, list):
                levels = [*filter(lambda x: x in noise.dims, total)]
            elif total is True:
                levels = [*filter(lambda x: x not in (self.kind, 'CHANNEL'),
                                  noise.dims)]
            noise = (1 / noise**2).sum(dim=levels)
            noise = 1 / numpy.sqrt(noise)

        noise.attrs = self.noise.attrs

        if channels == 1:
            return noise.squeeze('CHANNEL')
        return noise

    def redshift_range(self, low_frequency, high_frequency):
        """

        Parameters
        ----------
        low_frequency :

        high_frequency :


        Returns
        -------

        """

        zmax = high_frequency / self.frequency_range[0] - 1
        zmin = low_frequency / self.frequency_range[-1] - 1

        return zmin.clip(0), zmax

    def in_range(self, redshift, low_frequency, high_frequency):
        """

        Parameters
        ----------
        redshift :

        low_frequency :

        high_frequency :


        Returns
        -------

        """

        zmin, zmax = self.redshift_range(low_frequency, high_frequency)

        in_range = (zmin <= redshift) & (redshift <= zmax)
        return xarray.DataArray(in_range.astype(numpy.intp),
                                dims=self.kind)

    def get_frequency_response(self, spectral_index, channels=1):
        """

        Parameters
        ----------
        spectral_index : float

        channels : int
             (Default value = 1)

        Returns
        -------

        """

        nu = numpy.linspace(*self.frequency_range.value, channels + 1)
        nu = xarray.DataArray(nu, dims='CHANNEL')
        nu.attrs['unit'] = self.frequency_range.unit
        spec_idx = xarray.DataArray(spectral_index, dims=self.kind)
        nu_pow = nu**(1 + spec_idx)
        density_flux = nu_pow.diff('CHANNEL') / nu.diff('CHANNEL')
        density_flux.attrs['unit'] = 1 / nu.attrs['unit']

        if channels == 1:
            return density_flux.squeeze('CHANNEL')

        return density_flux.T

    def update(self, duration=0):
        """

        Parameters
        ----------
        duration :
             (Default value = 0)

        Returns
        -------

        """

        kw = {
            'alt': self.altaz.alt,
            'az': self.altaz.az,
            'obstime': self.altaz.obstime + duration
        }
        self.altaz = coordinates.AltAz(**kw)

    def copy(self):
        """ """

        return dill.copy(self)

    def select(self, idx, inplace=False):
        """

        Parameters
        ----------
        idx :

        inplace :
             (Default value = False)

        Returns
        -------

        """

        if not inplace:
            obs = self.copy()
            obs.select(idx, inplace=True)
            return obs

        self.response = self.response[idx]
        if hasattr(self, 'altaz'):
            self.altaz = self.altaz[idx]
        if hasattr(self, 'time_delay'):
            self.time_delay = self.time_delay[idx]


class Interferometry(Observation):
    """ """

    def __init__(self, obsi, obsj=None, degradation=None):

        kind, namei = obsi.response.dims

        if obsj is None:

            response = obsi.response
            beams = response.shape[-1]

            if beams > 1:

                noise = obsi.noise
                time_delay = getattr(obsi, 'time_delay', None)
                frequency_range = obsi.frequency_range
                width = frequency_range.diff()
                sampling_time = obsi.sampling_time

                coords = [
                    *map(lambda x: '{}x{}'.format(*x),
                         combinations(range(beams), 2))
                ]

                response = xarray.concat([
                    response[:, i] * response[:, j]
                    for i, j in combinations(range(beams), 2)
                ], dim=namei).assign_coords({namei: coords}).T
                response = numpy.sqrt(response / 2)

                unit = noise.attrs['unit']
                noise = xarray.concat([
                    noise[i] * noise[j]
                    for i, j in combinations(range(beams), 2)
                ], dim=namei).assign_coords({namei: coords})
                noise = numpy.sqrt(noise / 2)
                noise.attrs['unit'] = unit

                if time_delay is not None:
                    unit = time_delay.attrs['unit']
                    time_delay = xarray.concat([
                        time_delay[:, i] - time_delay[:, j]
                        for i, j in combinations(range(beams), 2)
                    ], dim=namei).assign_coords({namei: coords}).T
                    time_delay.attrs['unit'] = unit

            else:
                raise Exception('FRBlip does not compute self',
                                'correlations of single beams.')
        else:

            respi = obsi.response
            respj = obsj.response
            response = numpy.sqrt(respi * respj / 2)

            freqi = obsi.frequency_range
            freqj = obsj.frequency_range
            nu_1, nu_2 = numpy.column_stack([freqi, freqj])
            nu = units.Quantity([nu_1.max(), nu_2.min()])
            frequency_range = nu

            wi = freqi.diff()
            wj = freqj.diff()
            width = frequency_range.diff()

            qi = (wi / width).to(1)
            qj = (wj / width).to(1)

            dti = obsi.get_time_delay()
            dti.values = dti.values * dti.unit.to(units.ms)

            dtj = obsj.get_time_delay()
            dtj.values = dtj.values * dtj.unit.to(units.ms)

            dt = dti - dtj
            dt.attrs['unit'] = units.ms

            obs_ti = obsi.get_obstime()
            obs_tj = obsj.get_obstime()

            dt_obs = (obs_ti - obs_tj).to(units.ms)
            dt_obs = xarray.DataArray(dt_obs.value, dims=dt.dims[0],
                                      attrs={'unit': dt_obs.unit})

            time_delay = dt + dt_obs
            time_delay.attrs['unit'] = units.ms

            if (time_delay == 0).values.all():
                time_delay = None

            noisei = obsi.noise
            uniti = noisei.attrs['unit']

            noisej = obsj.noise
            unitj = noisej.attrs['unit']

            assert uniti == unitj, 'Incompatible noise units.'

            noisei = noisei * numpy.sqrt(qi)
            noisej = noisej * numpy.sqrt(qj)
            noise = numpy.sqrt(noisei * noisej / 2)
            noise.attrs['unit'] = uniti

            ti = obsi.sampling_time
            tj = obsj.sampling_time
            sampling_time = numpy.stack([ti, tj])
            sampling_time = sampling_time.max()

        Observation.__init__(self, response, noise, time_delay,
                             frequency_range, sampling_time)

        if degradation is not None:
            if isinstance(degradation, (float, int)):
                self.degradation = numpy.exp(-degradation**2 / 2)
                self.get_response = self.__degradation
            elif isinstance(degradation, tuple):
                self.amp, self.scale, self.power = degradation
                self.scale = (self.scale * units.MHz).to(1)
                self.get_response = self.__time_delay_degradation

    def __degradation(self, spectral_index, channels=1):
        response = Observation.get_response(self, spectral_index, channels)
        return self.degradation * response

    def __time_delay_degradation(self, spectral_index, channels=1):

        response = Observation.get_response(self, spectral_index, channels)
        log = (self.time_delay / self.scale)**self.power
        degradation = self.amp * numpy.exp(-log)
        return degradation * response
