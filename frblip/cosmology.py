import numpy
import pyccl
from astropy import units, constants

from functools import cached_property

from scipy.integrate import quad


class Cosmology(pyccl.Cosmology):

    DEFAULT_PARAMS = {
        'Planck_18': dict(
            Omega_c=0.261, Omega_b=0.049, h=0.6766, n_s=0.9665,
            sigma8=0.8102, Omega_k=0.0, Neff=3.046, T_CMB=2.7255,
            A_s=None, Omega_g=None, m_nu=0.0, m_nu_type=None,
            w0=-1.0, wa=0.0, bcm_log10Mc=14.079181246047625,
            bcm_etab=0.5, bcm_ks=55.0, mu_0=0.0, sigma_0=0.0, c1_mg=1.0,
            c2_mg=1.0, lambda_mg=0.0, z_mg=None, df_mg=None,
            transfer_function='boltzmann_camb',
            matter_power_spectrum='halofit',
            baryons_power_spectrum='nobaryons',
            mass_function='tinker10',
            halo_concentration='duffy2008',
            emulator_neutrinos='strict',
            extra_parameters=None
        )
    }

    def __init__(self, name='Planck_18', free_electron_bias='Takahashi2021',
                 **kwargs):

        kw = self.DEFAULT_PARAMS[name]
        kw.update(kwargs)

        super().__init__(**kw)
        self.__dict__.update(kw)

        self.H0 = 100 * self.h * units.km / units.s / units.Mpc
        self.m_nu = self.m_nu * units.eV
        self.T_CMB = self.T_CMB * units.K

        if isinstance(free_electron_bias, str):
            func_name = f'_{free_electron_bias.lower()}'
            self._free_electrons_bias = getattr(self, func_name)
        elif type(free_electron_bias) in (float, int):
            self._free_electrons_bias = self._constant_ebias
            self._eb = free_electron_bias

    def scale_factor(self, z):

        return 1 / (1 + z)

    def luminosity_distance(self, z):

        a = self.scale_factor(z)
        d = super().luminosity_distance(a)
        return d * units.Mpc

    def comoving_radial_distance(self, z):

        a = self.scale_factor(z)
        d = super().comoving_radial_distance(a)
        return d * units.Mpc

    def angular_diameter_distance(self, z):

        a = self.scale_factor(z)
        d = super().angular_diameter_distance(a)
        return d * units.Mpc

    def h_over_h0(self, z):

        a = self.scale_factor(z)
        return super().h_over_h0(a)

    def growth_factor(self, z):

        a = self.scale_factor(z)
        return super().growth_factor(a)

    def growth_rate(self, z):

        a = self.scale_factor(z)
        return super().growth_rate(a)

    def _takahashi2021(self, k, z=0):

        bs2 = 0.971 - 0.013 * z
        g = 1.91 - 0.59 * z + 0.10 * z**2
        ks = 4.36 - 3.24 * z + 3.10 * z**2 - 0.42 * z**3
        return bs2 / (1 + (k / ks)**g)

    def _constant_ebias(self, k, z=0):
        return self._eb

    def free_electrons_bias(self, k, z=0):

        ki = k / units.Mpc
        return self._ebias(ki, z)

    def _linear_matter_power(self, k, z=0):
        a = self.scale_factor(z)
        return super().linear_matter_power(k, a)

    def _nonlin_matter_power(self, k, z=0):
        a = self.scale_factor(z)
        return super().nonlin_matter_power(k, a)

    def _linear_power(self, k, z=0):
        a = self.scale_factor(z)
        return super().linear_power(k, a)

    def _nonlin_power(self, k, z=0):
        a = self.scale_factor(z)
        return super().nonlin_power(k, a)

    def linear_matter_power(self, k, z=0):

        k = k * units.Mpc
        P = self._linear_matter_power(k, z)
        return P * units.Mpc**3

    def nonlin_matter_power(self, k, z=0):

        k = k * units.Mpc
        P = self._nonlin_matter_power(k, z)
        return P * units.Mpc**3

    def linear_power(self, k, z=0):

        k = k * units.Mpc
        P = self._linear_power(k, z)
        return P * units.Mpc**3

    def nonlin_power(self, k, z=0):

        k = k * units.Mpc
        P = self._nonlin_power(k, z)
        return P * units.Mpc**3

    def _nonlin_electron_power(self, k, z=0):
        be = self._free_electrons_bias(k, z)
        P = self._nonlin_power(k, z)
        return be * P

    def nonlin_electron_power(self, k, z=0):

        ki = k * units.Mpc
        P = self._nonlin_electron_power(ki, z)
        return P * units.Mpc**3

    def _dm_igm_integral(self, z, kmin=0.0, kmax=numpy.inf):

        def _integrand(k):
            return k * self._nonlin_electron_power(k, z)

        integral, _ = quad(_integrand, kmin, kmax, limit=100, epsrel=1.49e-7)
        return integral

    def dm_igm_integral(self, z, kmin=0.0, kmax=numpy.inf, unit=units.Mpc):

        func = numpy.vectorize(self._dm_igm_integral,
                               excluded=['kmin', 'kmax'])
        return func(z, kmin, kmax) * unit / (2 * numpy.pi)

    def Hubble(self, z):

        E = self.h_over_h0(z)
        return self.H0 * E

    def differential_comoving_volume(self, z):

        r = self.comoving_radial_distance(z)
        Dh = constants.c / self.Hubble(z)
        dcoVol = (Dh * r**2).to(units.Mpc**3)
        return dcoVol / units.sr

    def star_formation_rate(self, z):

        num = 0.017 + 0.13 * z
        dem = 1 + (z / 3.3)**5.3
        unit = units.Msun / units.yr / units.Mpc**3
        sfr = (num / dem) * unit

        return sfr

    @cached_property
    def sfr0(self):

        return self.star_formation_rate(0)

    @cached_property
    def critical_density0(self):

        rho_c0 = 3 * self.H0**2 / (8 * numpy.pi * constants.G)
        return rho_c0.to(units.g / units.cm**3)

    @cached_property
    def hubble_distance(self):

        Dh = constants.c / self.H0
        return Dh.to(units.Mpc)

    @cached_property
    def baryon_number_density(self):

        Omega_b = self.Omega_b
        rho_c0 = self.critical_density0
        n = Omega_b * rho_c0 / constants.m_p
        return n.to(1 / units.cm**3)
