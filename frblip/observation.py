import numpy

from scipy.special import hyp1f1, erf

from itertools import combinations

from astropy.time import Time
from astropy import coordinates, units

from .utils import sub_dict, paired_shapes, squeeze_but_one


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


def specific_flux(spectral_index, frequency):
    """

    Parameters
    ----------
    spectral_index :

    frequency :


    Returns
    -------

    """

    nu = frequency[:, numpy.newaxis]
    si = spectral_index[numpy.newaxis]
    flux = (nu / units.MHz).to(1).value**si

    return flux.T


def density_flux(spectral_index, frequency):
    """

    Parameters
    ----------
    spectral_index :

    frequency :


    Returns
    -------

    """

    diff_nu = numpy.diff(frequency)
    nu = frequency[:, numpy.newaxis]
    si = spectral_index[numpy.newaxis]

    nup = (nu / units.MHz).to(1).value**(1 + si)
    flux = numpy.diff(nup, axis=0)
    return flux.T / diff_nu


def __intf(alpha, x):
    result = numpy.zeros_like(x)

    idx = alpha.ravel() == 0
    result[idx] = numpy.sin(x[idx])

    idx = alpha.ravel() != 0
    px = x[idx]
    ix = 1j * px

    a1 = alpha[idx] + 1
    h1f1 = hyp1f1(a1, a1 + 1, ix)
    result[idx] = (px**a1) * h1f1.real
    return result


@numpy.errstate(invalid='ignore')
def interferometry_density_flux(spectral_index, frequency, time_delays):
    """

    Parameters
    ----------
    spectral_index :

    frequency :

    time_delays :


    Returns
    -------

    """

    diff_nu = numpy.diff(frequency)
    si = spectral_index.reshape(-1, 1, 1)

    nu = numpy.atleast_3d(frequency)
    tau = time_delays[:, numpy.newaxis]
    k = 2 * numpy.pi * tau
    kx = (k * nu).to(1).value
    kn = (k * units.MHz).to(1).value
    intf = __intf(si, kx) / kn**(si + 1)
    intf = numpy.nansum(intf, axis=-1)

    flux = numpy.diff(intf, axis=-1)
    return flux / diff_nu


class Observation():
    """ """

    def __init__(self, response=None, noise=None, frequency_bands=None,
                 sampling_time=None, altaz=None, time_array=None):

        self.sampling_time = sampling_time

        self.response = response
        self.noise = noise
        self.frequency_bands = frequency_bands
        self.altaz = altaz

        if time_array is not None:
            self.time_array = time_array
            self.__n_array = time_array.shape[0]

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
        kwargs['obstime'] = Time(kwargs['obstime']).to_datetime()

        if location:
            kwargs['location'] = coordinates.EarthLocation(**location)

        self.altaz = coordinates.AltAz(**kwargs)

    def get_noise(self, channels=False):
        """

        Parameters
        ----------
        channels :
             (Default value = False)

        Returns
        -------

        """

        output = self.noise
        if not channels:
            inoise = (1 / output**2).sum(-1)
            output = 1 / numpy.sqrt(inoise)
        return output.squeeze()

    def pattern(self, spectral_index=None, channels=False):
        """

        Parameters
        ----------
        spectral_index :
             (Default value = None)
        channels :
             (Default value = False)

        Returns
        -------

        """
        return self.response

    def get_frequency(self, channels=False):
        """

        Parameters
        ----------
        channels :
             (Default value = False)

        Returns
        -------

        """
        if channels:
            return self.frequency_bands
        return self.frequency_bands[[0, -1]]

    def get_response(self, spectral_index, channels=False):
        """

        Parameters
        ----------
        spectral_index :

        channels :
             (Default value = False)

        Returns
        -------

        """
        nu = self.get_frequency(channels)
        S = density_flux(spectral_index, nu)
        pattern = self.pattern()
        response = pattern[..., numpy.newaxis] * S[:, numpy.newaxis]
        return squeeze_but_one(response)

    def time_difference(self):
        """ """

        if hasattr(self, 'time_array'):
            ran = range(self.__n_array)
            comb = combinations(ran, 2)
            i, j = numpy.array([*comb]).T
            dt = self.time_array[i] - self.time_array[j]
            return dt.to(units.ns).T
        return None

    def to_dict(self, flag=''):
        """

        Parameters
        ----------
        flag :
             (Default value = '')

        Returns
        -------

        """

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
            obstime = obstime.to_datetime().astype(numpy.str_)
            out_dict['obstime'] = obstime

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
        """

        Parameters
        ----------
        kwargs :


        Returns
        -------

        """

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


def time_difference(obsi, obsj):
    """

    Parameters
    ----------
    obsi :

    obsj :


    Returns
    -------

    """

    ti = obsi.altaz.obstime
    tj = obsj.altaz.obstime

    n_frb = numpy.unique([ti.size, tj.size]).item(0)

    dt = (tj - ti).to(units.ms)

    t_arrayi = getattr(obsi, 'time_array', numpy.zeros((1, n_frb)))
    t_arrayj = getattr(obsj, 'time_array', numpy.zeros((1, n_frb)))

    Dt = t_arrayj[numpy.newaxis] - t_arrayi[:, numpy.newaxis]
    dt = (dt - Dt).reshape((-1, n_frb))

    return dt.T


def cross_response(obsi, obsj):
    """

    Parameters
    ----------
    obsi :

    obsj :


    Returns
    -------

    """

    respi = obsi.response[..., numpy.newaxis]
    respj = obsj.response[:, numpy.newaxis]
    return numpy.sqrt(respi * respj / 2)


def cross_noise(obsi, obsj):
    """

    Parameters
    ----------
    obsi :

    obsj :


    Returns
    -------

    """

    noisei = obsi.noise[:, numpy.newaxis]
    noisej = obsj.noise[numpy.newaxis]
    return numpy.sqrt(noisei * noisej / 2).to(units.Jy)


class Interferometry():
    """ """

    def __init__(self, *observations, time_delay=False):

        n_scopes = len(observations)
        freqs = numpy.stack([
            observation.frequency_bands
            for observation in observations
        ])
        self.frequency_bands = numpy.unique(freqs, axis=0).ravel()

        sampling_time = numpy.stack([
            observation.sampling_time
            for observation in observations
        ])
        self.sampling_time = numpy.unique(sampling_time).item()

        n_beams = numpy.array([
            observation._Observation__n_beam
            for observation in observations
        ]).ravel()

        shapes = paired_shapes(n_beams)
        xshapes = shapes[n_scopes:]
        shapes = shapes[:n_scopes]

        self.response = [
            observation.response.reshape((-1, *shape)) / numpy.sqrt(2)
            for shape, observation in zip(shapes, observations)
            if hasattr(observation, 'time_array')
        ]

        self.response += [
            cross_response(*obsij).reshape((-1, *shape))
            for shape, obsij in zip(xshapes, combinations(observations, 2))
        ]

        self.noises = [
            observation.noise.reshape((*shape, -1)) / numpy.sqrt(2)
            for shape, observation in zip(shapes, observations)
            if hasattr(observation, 'time_array')
        ]

        self.noises += [
            cross_noise(*obsij).reshape((*shape, -1))
            for shape, obsij in zip(xshapes, combinations(observations, 2))
        ]

        self.time_delays = [
            observation.time_difference()
            for observation in observations
            if hasattr(observation, 'time_array')
        ]

        self.time_delays += [
            time_difference(*obsij)
            for obsij in combinations(observations, 2)
        ]

        self.__pairs = numpy.array([
            time_delay.shape[-1] if time_delay.ndim > 1 else 1
            for time_delay in self.time_delays
        ])

        if time_delay:
            self.get_response = self.__time_delay
        else:
            self.get_response = self.__no_time_delay

            self.response = sum([
                pair * response
                for pair, response in zip(self.__pairs, self.response)
            ], numpy.zeros(()))

            self.response = squeeze_but_one(self.response)

    def get_noise(self, channels=False):
        """

        Parameters
        ----------
        channels :
             (Default value = False)

        Returns
        -------

        """

        output = [
            pairs / noise**2
            for pairs, noise in zip(self.__pairs, self.noises)
        ]
        output = sum(output, numpy.zeros(()))

        if channels:
            output = 1 / numpy.sqrt(output)
        else:
            output = 1 / numpy.sqrt(output.sum(-1))

        return output.squeeze()

    def get_frequency(self, channels=False):
        """

        Parameters
        ----------
        channels :
             (Default value = False)

        Returns
        -------

        """
        nu = self.frequency_bands
        return nu if channels else nu[[0, -1]]

    def pattern(self, spectral_index, channels=False):
        """

        Parameters
        ----------
        spectral_index :

        channels :
             (Default value = False)

        Returns
        -------

        """

        response = self.get_response(spectral_index, channels)

        nu = self.get_frequency(channels)

        sflux = density_flux(spectral_index, nu)
        ndims = tuple(range(1, response.ndim - 1))
        sflux = numpy.expand_dims(sflux, ndims)

        pattern = response / sflux

        return pattern.value

    def __no_time_delay(self, spectral_index, channels=False):

        nu = self.get_frequency(channels)
        sflux = density_flux(spectral_index, nu)
        return self.response * sflux

    def __time_delay(self, spectral_index, channels=False):

        nu = self.get_frequency(channels)

        interfs = [
            interferometry_density_flux(spectral_index, nu, time_delay)
            for time_delay in self.time_delays
        ]

        unit = [interf.unit for interf in interfs]
        unit = 1 * numpy.unique(unit).item()

        dims = [
            tuple(range(1, response.ndim))
            for response in self.response
        ]

        values = [
            numpy.expand_dims(interf, axis=dim).value
            for interf, dim in zip(interfs, dims)
        ]

        value = sum([
            resp[..., numpy.newaxis] * value
            for resp, value in zip(self.response, values)
        ], numpy.zeros(()))

        response = value * unit
        return squeeze_but_one(response)
