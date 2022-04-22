import numpy
import xarray

from itertools import combinations

from scipy.special import hyp1f1, erf

from astropy.time import Time
from astropy import coordinates, units

from .utils import sub_dict, squeeze_but_one


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


def density_flux(spectral_index, frequency):
    """

    Parameters
    ----------
    spectral_index :

    frequency :


    Returns
    -------

    """
    si = spectral_index.reshape(-1, 1)
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

    def __init__(self, response=None, noise=None, channels=None,
                 frequency_range=None, sampling_time=None,
                 altaz=None, time_array=None):

        self.sampling_time = sampling_time

        self.response = response
        self.noise = noise
        self.frequency_range = frequency_range
        self.channels = channels
        self.altaz = altaz
        self.kind = response.dims[0]

        self.width = self.frequency_range.diff()

        if time_array is not None:
            self.time_array = time_array
            self.__n_array = time_array.shape[0]

        self.__set_response()

    def __set_response(self):

        noise = self.noise is not None
        response = self.response is not None

        if response and noise:
            self.__n_beam = self.response.shape[1:]
            self.__n_telescopes = numpy.size(self.__n_beam)

    def __set_coordinates(self):

        params = self.__dict__

        location = sub_dict(params, pop=True, keys=['lon', 'lat', 'height'])
        kwargs = sub_dict(params, pop=True, keys=['alt', 'az', 'obstime'])
        kwargs['obstime'] = Time(kwargs['obstime']).to_datetime()

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
            return numpy.linspace(*self.frequency_range, self.channels + 1)
        return self.frequency_range

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
        nu = numpy.linspace(*self.frequency_range, channels + 1)
        spec_idx = spectral_index.reshape(-1, 1)
        nu_pow = (nu / units.MHz)**(1 + spec_idx)
        density_flux = nu_pow.diff(axis=-1) / nu.diff()
        density_flux = xarray.DataArray(density_flux * self.width,
                                        dims=(self.kind, 'CHANNEL'))
        return self.response * density_flux

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

    t_arrayi = getattr(obsi, 'time_array', numpy.zeros((n_frb, 1)))
    t_arrayj = getattr(obsj, 'time_array', numpy.zeros((n_frb, 1)))

    Dt = t_arrayj[:, numpy.newaxis] - t_arrayi[..., numpy.newaxis]
    dt = dt[..., numpy.newaxis] - Dt.reshape(n_frb, -1)

    return squeeze_but_one(dt)


class Interferometry():
    """ """

    def __init__(self, obsi, obsj=None, time_delay=False):

        if time_delay:
            self.get_response = self.__time_delay
        else:
            self.get_response = self.__no_time_delay

        self.kind, namei = obsi.response.dims

        if obsj is None:

            response = obsi.response
            beams = response.shape[-1]

            if beams > 1:

                noise = obsi.noise
                self.frequency_range = obsi.frequency_range
                self.width = self.frequency_range.diff()
                self.channels = obsi.channels
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

            else:
                raise Exception('FRBlip does not compute self',
                                'correlations of single beams.')
        else:

            respi = obsi.response
            respj = obsj.response
            self.response = numpy.sqrt(respi * respj / 2)
            self.response = squeeze_but_one(self.response)

            self.time_delay = time_difference(obsi, obsj)
            self.pairs = self.time_delay.shape[-1]

            if time_delay:
                self.get_response = self.__time_delay
            else:
                self.get_response = self.__no_time_delay
                self.response = self.pairs * self.response

            freqi = obsi.frequency_range
            freqj = obsj.frequency_range
            nu_1, nu_2 = numpy.column_stack([freqi, freqj])
            self.frequency_range = units.Quantity([nu_1.max(), nu_2.min()])

            wi = freqi.diff()
            wj = freqj.diff()
            self.width = self.frequency_range.diff()

            noisei = obsi.noise
            noisej = obsj.noise
            noise = 0.5 * noisei * noisej * wi * wj
            self.noise = numpy.sqrt(noise / self.pairs)
            self.noise = self.noise / self.width

            ti = obsi.sampling_time
            tj = obsj.sampling_time
            sampling_time = numpy.stack([ti, tj])
            self.sampling_time = sampling_time.max()

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
            return numpy.linspace(*self.frequency_range, self.channels + 1)
        return self.frequency_range

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

    def __no_time_delay(self, spectral_index, channels=1):

        nu = numpy.linspace(*self.frequency_range, channels + 1)
        spec_idx = spectral_index.reshape(-1, 1)
        nu_pow = (nu / units.MHz)**(1 + spec_idx)
        density_flux = nu_pow.diff(axis=-1) / nu.diff()
        density_flux = xarray.DataArray(density_flux * self.width,
                                        dims=(self.kind, 'CHANNEL'))
        return self.response * density_flux

    def __time_delay(self, spectral_index, channels=False):

        nu = self.get_frequency(channels)

        interf = interferometry_density_flux(spectral_index, nu,
                                             self.time_delay)
        return self.response * interf
