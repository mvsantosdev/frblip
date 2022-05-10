import numpy
import xarray

from itertools import combinations

from scipy.special import erf
from astropy import coordinates, units


def disperse(nu, DM):
    D = DM * units.cm**3 / units.pc
    return 4.15 * units.ms * D * (units.MHz / nu)**2


def gaussian(t, w, t0=0.0):
    z = (t - t0) / w
    return numpy.exp(- z**2 / 2)


def scattered_gaussian(t, w, ts, t0=0.0):

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

    def __init__(self, response=None, noise=None, altaz=None,
                 time_array=None, frequency_range=None,
                 sampling_time=None):

        self.response = response
        self.noise = noise
        self.altaz = altaz
        self.time_array = time_array
        self.frequency_range = frequency_range
        self.sampling_time = sampling_time

        self.kind = response.dims[0]
        self.width = self.frequency_range.diff()

    def __set_coordinates(self):

        params = self.__dict__
        location = {key: params.pop(key) for key in ('lon', 'lat', 'height')}
        kwargs = {key: params.pop(key) for key in ('alt', 'az', 'obstime')}

        if location:
            kwargs['location'] = coordinates.EarthLocation(**location)

        self.altaz = coordinates.AltAz(**kwargs)

    def get_noise(self, channels=1):
        """

        Parameters
        ----------
        channels :
             (Default value = False)

        Returns
        -------

        """
        noise = numpy.full(channels, numpy.sqrt(channels))
        noise = self.noise * xarray.DataArray(noise, dims='CHANNEL')
        return noise

    def get_response(self, spectral_index, channels=1):
        """

        Parameters
        ----------
        spectral_index :

        channels :
             (Default value = False)

        Returns
        -------

        """
        nu = (self.frequency_range / units.MHz).to(1)
        nu = numpy.linspace(*nu, channels + 1)
        nu = xarray.DataArray(nu, dims='CHANNEL')
        spec_idx = xarray.DataArray(spectral_index, dims=self.kind)
        nu_pow = nu**(1 + spec_idx)
        density_flux = nu_pow.diff('CHANNEL') / nu.diff('CHANNEL')
        density_flux = density_flux * (self.width / units.MHz).to(1)
        return self.response * density_flux

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            return self.select(idx)
        idx = numpy.array(idx)
        return self.select(idx)

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
        response = self.response[idx]
        altaz = getattr(self, 'altaz', None)
        altaz = altaz[idx] if altaz else None
        if not inplace:
            output = Observation.__new__(Observation)
            output.__dict__.update(self.__dict__)
            output.response = response
            return output
        self.response = response


class Interferometry(Observation):
    """ """

    def __init__(self, obsi, obsj=None, degradation=None):

        self.kind, namei = obsi.response.dims

        if obsj is None:

            response = obsi.response
            beams = response.shape[-1]

            if beams > 1:

                noise = obsi.noise
                time_array = obsi.time_array
                self.frequency_range = obsi.frequency_range
                self.width = self.frequency_range.diff()
                self.sampling_time = obsi.sampling_time

                response = numpy.column_stack([
                    response[:, i] * response[:, j]
                    for i, j in combinations(range(beams), 2)
                ])
                dims = self.kind, namei
                response = xarray.DataArray(response, dims=dims)
                self.response = numpy.sqrt(response / 2)

                noise = numpy.array([
                    noise[i] * noise[j]
                    for i, j in combinations(range(beams), 2)
                ])
                noise = xarray.DataArray(noise, dims=namei)
                self.noise = numpy.sqrt(noise / 2)

                time_delay = numpy.column_stack([
                    time_array[:, i] - time_array[:, j]
                    for i, j in combinations(range(beams), 2)
                ])
                self.time_delay = xarray.DataArray(time_delay, dims=dims)

            else:
                raise Exception('FRBlip does not compute self',
                                'correlations of single beams.')
        else:

            freqi = obsi.frequency_range
            freqj = obsj.frequency_range
            nu_1, nu_2 = numpy.column_stack([freqi, freqj])
            nu = units.Quantity([nu_1.max(), nu_2.min()])
            self.frequency_range = nu

            wi = freqi.diff()
            wj = freqj.diff()
            self.width = self.frequency_range.diff()

            qi = (wi / self.width).to(1)
            qj = (wj / self.width).to(1)

            respi = obsi.response * qi
            respj = obsj.response * qj
            self.response = numpy.sqrt(respi * respj / 2)
            sign = numpy.sign(respi)
            self.response = sign * self.response

            dt = obsi.time_array - obsj.time_array
            Dt = obsi.altaz.obstime - obsj.altaz.obstime
            Dt = (Dt * units.MHz).to(1)
            Dt = xarray.DataArray(Dt, dims='FRB')
            self.time_delay = dt + Dt

            noisei = obsi.noise
            noisej = obsj.noise
            noise = noisei * noisej * qi * qj
            self.noise = numpy.sqrt(noise / 2)

            ti = obsi.sampling_time
            tj = obsj.sampling_time
            sampling_time = numpy.stack([ti, tj])
            self.sampling_time = sampling_time.max()

        if degradation is not None:
            if isinstance(degradation, (float, int)):
                self.degradation = 1 + numpy.exp(-degradation**2 / 2)
                self.get_response = self.__degradation
            elif isinstance(degradation, tuple):
                self.amp, self.scale, self.power = degradation
                self.scale = (self.scale * units.MHz).to(1)
                self.get_response = self.__time_delay_degradation

    def __degradation(self, spectral_index, channels=1):
        response = super().get_response(spectral_index, channels)
        return self.degradation * response

    def __time_delay_degradation(self, spectral_index, channels=1):

        response = super().get_response(spectral_index, channels)
        log = (self.time_delay / self.scale)**self.power
        degradation = 1 + self.amp * numpy.exp(-log)
        return degradation * response
