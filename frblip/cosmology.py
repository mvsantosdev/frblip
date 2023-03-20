from functools import cached_property

import numpy
import pyccl
from astropy import constants, units
from scipy.integrate import quad

from .decorators import default_units, from_source


class Cosmology(pyccl.Cosmology):

    DEFAULT_PARAMS = {
        'Planck_18': dict(
            Omega_c=0.261,
            Omega_b=0.049,
            h=0.6766,
            n_s=0.9665,
            sigma8=0.8102,
            Omega_k=0.0,
            Neff=3.046,
            T_CMB=2.7255,
            A_s=None,
            Omega_g=None,
            m_nu=0.0,
            m_nu_type=None,
            w0=-1.0,
            wa=0.0,
            bcm_log10Mc=14.079181246047625,
            bcm_etab=0.5,
            bcm_ks=55.0,
            mu_0=0.0,
            sigma_0=0.0,
            c1_mg=1.0,
            c2_mg=1.0,
            lambda_mg=0.0,
            z_mg=None,
            df_mg=None,
            transfer_function='boltzmann_camb',
            matter_power_spectrum='halofit',
            baryons_power_spectrum='nobaryons',
            mass_function='tinker10',
            halo_concentration='duffy2008',
            emulator_neutrinos='strict',
            extra_parameters=None,
        )
    }

    @from_source(default_dict=DEFAULT_PARAMS)
    def __init__(
        self,
        source: str = 'Planck_18',
        free_electron_bias: str = 'Takahashi2021',
        Omega_c: float = 0.261,
        Omega_b: float = 0.049,
        h: float = 0.6766,
        n_s: float = 0.9665,
        sigma8: float | None = 0.8102,
        Omega_k: float | None = 0.0,
        Neff: float | None = 3.046,
        T_CMB: float = 2.7255,
        A_s: float | None = None,
        Omega_g: float | None = None,
        m_nu: float | None = 0.0,
        m_nu_type: float | None = None,
        w0: float | None = -1.0,
        wa: float | None = 0.0,
        bcm_log10Mc: float | None = 14.079181246047625,
        bcm_etab: float | None = 0.5,
        bcm_ks: float | None = 55.0,
        mu_0: float | None = 0.0,
        sigma_0: float | None = 0.0,
        c1_mg: float | None = 1.0,
        c2_mg: float | None = 1.0,
        lambda_mg: float | None = 0.0,
        z_mg: float | None = None,
        df_mg: float | None = None,
        transfer_function: str | None = 'boltzmann_camb',
        matter_power_spectrum: str | None = 'halofit',
        baryons_power_spectrum: str | None = 'nobaryons',
        mass_function: str | None = 'tinker10',
        halo_concentration: str | None = 'duffy2008',
        emulator_neutrinos: str | None = 'strict',
        extra_parameters: dict | None = None,
    ):

        super().__init__(
            Omega_c,
            Omega_b,
            h,
            n_s,
            sigma8,
            A_s,
            Omega_k,
            Omega_g,
            Neff,
            m_nu,
            m_nu_type,
            w0,
            wa,
            T_CMB,
            bcm_log10Mc,
            bcm_etab,
            bcm_ks,
            mu_0,
            sigma_0,
            c1_mg,
            c2_mg,
            lambda_mg,
            z_mg,
            df_mg,
            transfer_function,
            matter_power_spectrum,
            baryons_power_spectrum,
            mass_function,
            halo_concentration,
            emulator_neutrinos,
            extra_parameters,
        )

        self._load_params(h, m_nu, T_CMB, Omega_b)

        if isinstance(free_electron_bias, str):
            func_name = f'_{free_electron_bias.lower()}'
            self._free_electrons_bias = getattr(self, func_name)
        elif type(free_electron_bias) in (float, int):
            self._free_electrons_bias = self._constant_ebias
            self._eb = free_electron_bias

    @default_units(h='km s^-1 Mpc^-1', m_nu='eV', T_CMB='K')
    def _load_params(
        self,
        h: float | units.Quantity,
        m_nu: float | units.Quantity,
        T_CMB: float | units.Quantity,
        Omega_b: float,
    ):
        self.H0 = 100 * h
        self.m_nu = m_nu
        self.T_CMB = T_CMB
        self.Omega_b = Omega_b

    def scale_factor(self, z: float | numpy.ndarray) -> float | numpy.ndarray:

        return 1 / (1 + z)

    @default_units('Mpc')
    def luminosity_distance(self, z: float | numpy.ndarray) -> units.Quantity:

        a = self.scale_factor(z)
        return super().luminosity_distance(a)

    @default_units('Mpc')
    def comoving_radial_distance(
        self, z: float | numpy.ndarray
    ) -> units.Quantity:

        a = self.scale_factor(z)
        return super().comoving_radial_distance(a)

    @default_units('Mpc')
    def angular_diameter_distance(
        self, z: float | numpy.ndarray
    ) -> units.Quantity:

        a = self.scale_factor(z)
        return super().angular_diameter_distance(a)

    def h_over_h0(self, z: float | numpy.ndarray) -> units.Quantity:

        a = self.scale_factor(z)
        return super().h_over_h0(a)

    def growth_factor(self, z: float | numpy.ndarray) -> units.Quantity:

        a = self.scale_factor(z)
        return super().growth_factor(a)

    def growth_rate(self, z: float | numpy.ndarray) -> units.Quantity:

        a = self.scale_factor(z)
        return super().growth_rate(a)

    def _takahashi2021(
        self,
        k: float | numpy.ndarray | units.Quantity,
        z: float | numpy.ndarray = 0,
    ) -> units.Quantity:

        bs2 = 0.971 - 0.013 * z
        g = 1.91 - 0.59 * z + 0.10 * z**2
        ks = 4.36 - 3.24 * z + 3.10 * z**2 - 0.42 * z**3
        return bs2 / (1 + (k / ks) ** g)

    def _constant_ebias(
        self,
        k: float | numpy.ndarray | units.Quantity,
        z: float | numpy.ndarray = 0,
    ) -> float:
        return self._eb

    @default_units(k='Mpc^-1')
    def free_electrons_bias(
        self,
        k: float | numpy.ndarray | units.Quantity,
        z: float | numpy.ndarray = 0,
    ) -> numpy.ndarray | float:

        return self._ebias(k.value, z)

    def _linear_matter_power(
        self,
        k: float | numpy.ndarray,
        z: float | numpy.ndarray = 0,
    ) -> units.Quantity:

        a = self.scale_factor(z)
        return super().linear_matter_power(k, a)

    def _nonlin_matter_power(
        self,
        k: float | numpy.ndarray,
        z: float | numpy.ndarray = 0,
    ) -> units.Quantity:

        a = self.scale_factor(z)
        return super().nonlin_matter_power(k, a)

    def _linear_power(
        self,
        k: float | numpy.ndarray,
        z: float | numpy.ndarray = 0,
    ) -> units.Quantity:

        a = self.scale_factor(z)
        return super().linear_power(k, a)

    def _nonlin_power(
        self,
        k: float | numpy.ndarray,
        z: float | numpy.ndarray = 0,
    ) -> units.Quantity:

        a = self.scale_factor(z)
        return super().nonlin_power(k, a)

    def _nonlin_electron_power(
        self,
        k: float | numpy.ndarray | units.Quantity,
        z: float | numpy.ndarray = 0,
    ) -> units.Quantity:

        be = self._free_electrons_bias(k, z)
        P = self._nonlin_power(k, z)
        return be * P

    @default_units('Mpc^3', k='Mpc^-1')
    def linear_matter_power(
        self,
        k: float | numpy.ndarray | units.Quantity,
        z: float | numpy.ndarray = 0,
    ) -> units.Quantity:

        return self._linear_matter_power(k.value, z)

    @default_units('Mpc^3', k='Mpc^-1')
    def nonlin_matter_power(
        self,
        k: float | numpy.ndarray | units.Quantity,
        z: float | numpy.ndarray = 0,
    ) -> units.Quantity:

        return self._nonlin_matter_power(k.value, z)

    @default_units('Mpc^3', k='Mpc^-1')
    def linear_power(
        self,
        k: float | numpy.ndarray | units.Quantity,
        z: float | numpy.ndarray = 0,
    ) -> units.Quantity:

        return self._linear_power(k.value, z)

    @default_units('Mpc^3', k='Mpc^-1')
    def nonlin_power(
        self,
        k: float | numpy.ndarray | units.Quantity,
        z: float | numpy.ndarray = 0,
    ) -> units.Quantity:

        return self._nonlin_power(k.value, z)

    @default_units('Mpc^3', k='Mpc^-1')
    def nonlin_electron_power(
        self,
        k: float | numpy.ndarray | units.Quantity,
        z: float | numpy.ndarray = 0,
    ) -> units.Quantity:

        return self._nonlin_electron_power(k.value, z)

    def _dm_igm_integral(
        self,
        z: float | numpy.ndarray,
        kmin: float = 0.0,
        kmax: float = numpy.inf,
    ) -> numpy.ndarray:
        def _integrand(k):
            return k * self._nonlin_electron_power(k, z)

        integral, _ = quad(_integrand, kmin, kmax, limit=100, epsrel=1.49e-7)
        return integral

    def dm_igm_integral(
        self,
        z: float | numpy.ndarray,
        kmin: float = 0.0,
        kmax: float = numpy.inf,
        unit: units.Unit = units.Mpc,
    ) -> units.Quantity:

        func = numpy.vectorize(
            self._dm_igm_integral, excluded=['kmin', 'kmax']
        )
        return func(z, kmin, kmax) * unit / (2 * numpy.pi)

    @default_units('km s^-1 Mpc^-1')
    def Hubble(self, z: numpy.ndarray | float) -> units.Quantity:

        E = self.h_over_h0(z)
        return self.H0 * E

    @default_units('Mpc^3 sr^-1')
    def differential_comoving_volume(
        self, z: numpy.ndarray | float
    ) -> units.Quantity:

        r = self.comoving_radial_distance(z)
        Dh = constants.c / self.Hubble(z)
        return Dh * r**2 / units.sr

    @default_units('Msun yr^-1 Mpc^-3')
    def star_formation_rate(self, z: numpy.ndarray | float) -> units.Quantity:

        num = 0.017 + 0.13 * z
        dem = 1 + (z / 3.3) ** 5.3
        return num / dem

    @cached_property
    def sfr0(self) -> units.Quantity:

        return self.star_formation_rate(0)

    @cached_property
    @default_units('g cm^-3')
    def critical_density0(self) -> units.Quantity:

        return 3 * self.H0**2 / (8 * numpy.pi * constants.G)

    @cached_property
    @default_units('Mpc')
    def hubble_distance(self) -> units.Quantity:

        return constants.c / self.H0

    @cached_property
    @default_units('cm^-3')
    def baryon_number_density(self) -> units.Quantity:

        Omega_b = self.Omega_b
        rho_c0 = self.critical_density0
        return Omega_b * rho_c0 / constants.m_p
