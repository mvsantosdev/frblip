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

    def __init__(self, altaz=None, peak_density_flux=None,
                 response=None, noise=None, time_delay=None,
                 frequency_range=None, sampling_time=None):

        self.altaz = altaz
        self.peak_density_flux = peak_density_flux
        self.response = response
        self.noise = noise
        if time_delay is not None:
            self.time_delay = time_delay
        self.frequency_range = frequency_range
        self.sampling_time = sampling_time

        self.kind = response.dims[0]
        self.width = self.frequency_range.diff()

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
        noise.attrs = self.noise.attrs
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
        density_flux = self.response * self.peak_density_flux * density_flux
        density_flux.attrs = self.peak_density_flux.attrs
        return density_flux.squeeze()

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
        peak_density_flux = self.peak_density_flux[idx]
        if not inplace:
            output = Observation.__new__(Observation)
            output.__dict__.update(self.__dict__)
            output.response = response
            output.altaz = altaz
            output.peak_density_flux = peak_density_flux
            return output
        self.response = response


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
                speak = obsi.peak_density_flux

                response = xarray.concat([
                    response[:, i] * response[:, j]
                    for i, j in combinations(range(beams), 2)
                ], dim=namei).T
                response = numpy.sqrt(response / 2)

                unit = noise.attrs['unit']
                noise = xarray.concat([
                    noise[i] * noise[j]
                    for i, j in combinations(range(beams), 2)
                ], dim=namei)
                noise = numpy.sqrt(noise / 2)
                noise.attrs['unit'] = unit

                if time_delay is not None:
                    unit = time_delay.attrs['unit']
                    time_delay = xarray.concat([
                        time_delay[:, i] - time_delay[:, j]
                        for i, j in combinations(range(beams), 2)
                    ], dim=namei).T
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

            speaki = obsi.peak_density_flux
            speakj = obsj.peak_density_flux

            speaki = speaki * wi
            speakj = speakj * wj

            assert numpy.isclose(speaki, speakj, rtol=1e-15).all(), \
                   'The observations do not correspond to same dataset'

            speak = speaki / width

            Dti = obsi.altaz.obstime
            Dtj = obsj.altaz.obstime
            Dt = (Dti - Dtj).to(units.ms)

            dti = getattr(obsi, 'time_delay', 0)
            if hasattr(dti, 'attrs'):
                dti = dti * dti.attrs.get('unit', 1).to(Dt.unit)

            dtj = getattr(obsj, 'time_delay', 0)
            if hasattr(dtj, 'attrs'):
                dtj = dtj * dtj.attrs.get('unit', 1).to(Dt.unit)

            dt = dti - dtj

            time_delay = dt + xarray.DataArray(Dt, dims=dt.dims[0])
            time_delay.attrs['unit'] = Dt.unit

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

        Observation.__init__(self, peak_density_flux=speak,
                             response=response, noise=noise,
                             time_delay=time_delay,
                             frequency_range=frequency_range,
                             sampling_time=sampling_time)

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
