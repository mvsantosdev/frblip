import numpy
import pyccl
from astropy import units, constants

from functools import cached_property

from scipy.integrate import quad


builtin = {
    'Planck_18': {
        'Omega_c': 0.261, 'Omega_b': 0.049, 'h': 0.6766, 'n_s': 0.9665,
        'sigma8': 0.8102, 'Omega_k': 0.0, 'Neff': 3.046, 'T_CMB': 2.7255
    }
}


class Cosmology(pyccl.Cosmology):
    """ """

    def __init__(self, Omega_c=None, Omega_b=None, h=None,
                 n_s=None, sigma8=None, A_s=None, Omega_k=0.0,
                 Omega_g=None, Neff=3.046, m_nu=0.0, m_nu_type=None,
                 w0=-1.0, wa=0.0, T_CMB=2.725, bcm_log10Mc=14.079181246047625,
                 bcm_etab=0.5, bcm_ks=55.0, mu_0=0.0, sigma_0=0.0, c1_mg=1.0,
                 c2_mg=1.0, lambda_mg=0.0, z_mg=None, df_mg=None,
                 transfer_function='boltzmann_camb',
                 matter_power_spectrum='halofit',
                 baryons_power_spectrum='nobaryons',
                 mass_function='tinker10',
                 halo_concentration='duffy2008',
                 emulator_neutrinos='strict',
                 extra_parameters=None):

        self.Omega_c = Omega_c
        self.Omega_b = Omega_b
        self.h = h
        self.H0 = 100 * h * units.km / units.s / units.Mpc
        self.n_s = n_s
        self.sigma8 = sigma8
        self.A_s = A_s
        self.Omega_k = Omega_k
        self.Omega_g = Omega_g
        self.Neff = Neff
        self.m_nu = m_nu * units.eV
        self.m_nu_type = m_nu_type
        self.w0 = w0
        self.wa = wa
        self.T_CMB = T_CMB * units.K

        super().__init__(Omega_c, Omega_b, h, n_s, sigma8, A_s, Omega_k,
                         Omega_g, Neff, m_nu, m_nu_type, w0, wa, T_CMB,
                         bcm_log10Mc, bcm_etab, bcm_ks, mu_0, sigma_0,
                         c1_mg, c2_mg, lambda_mg, z_mg, df_mg,
                         transfer_function, matter_power_spectrum,
                         baryons_power_spectrum, mass_function,
                         halo_concentration, emulator_neutrinos,
                         extra_parameters)

    def scale_factor(self, z):
        """

        Parameters
        ----------
        z :

        Returns
        -------

        """
        return 1 / (1 + z)

    def luminosity_distance(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        a = self.scale_factor(z)
        d = super().luminosity_distance(a)
        return d * units.Mpc

    def comoving_radial_distance(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        a = self.scale_factor(z)
        d = super().comoving_radial_distance(a)
        return d * units.Mpc

    def angular_diameter_distance(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        a = self.scale_factor(z)
        d = super().angular_diameter_distance(a)
        return d * units.Mpc

    def h_over_h0(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        a = self.scale_factor(z)
        return super().h_over_h0(a)

    def growth_factor(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        a = self.scale_factor(z)
        return super().growth_factor(a)

    def growth_rate(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        a = self.scale_factor(z)
        return super().growth_rate(a)

    def __free_electrons_bias(self, k, z=0):

        bs2 = 0.971 - 0.013 * z
        g = 1.91 - 0.59 * z + 0.10 * z**2
        ks = 4.36 - 3.24 * z + 3.10 * z**2 - 0.42 * z**3
        return bs2 / (1 + (k / ks)**g)

    def free_electrons_bias(self, k, z=0):
        """

        Parameters
        ----------
        k :

        z :
             (Default value = 0)

        Returns
        -------

        """
        ki = k / units.Mpc
        return self.__free_electrons_bias(ki, z)

    def __linear_matter_power(self, k, z=0):
        a = self.scale_factor(z)
        return super().linear_matter_power(k, a)

    def __nonlin_matter_power(self, k, z=0):
        a = self.scale_factor(z)
        return super().nonlin_matter_power(k, a)

    def __linear_power(self, k, z=0):
        a = self.scale_factor(z)
        return super().linear_power(k, a)

    def __nonlin_power(self, k, z=0):
        a = self.scale_factor(z)
        return super().nonlin_power(k, a)

    def linear_matter_power(self, k, z=0):
        """

        Parameters
        ----------
        k :

        z :
             (Default value = 0)

        Returns
        -------

        """
        k = k * units.Mpc
        P = self.__linear_matter_power(k, z)
        return P * units.Mpc**3

    def nonlin_matter_power(self, k, z=0):
        """

        Parameters
        ----------
        k :

        z :
             (Default value = 0)

        Returns
        -------

        """
        k = k * units.Mpc
        P = self.__nonlin_matter_power(k, z)
        return P * units.Mpc**3

    def linear_power(self, k, z=0):
        """

        Parameters
        ----------
        k :

        z :
             (Default value = 0)

        Returns
        -------

        """
        k = k * units.Mpc
        P = self.__linear_power(k, z)
        return P * units.Mpc**3

    def nonlin_power(self, k, z=0):
        """

        Parameters
        ----------
        k :

        z :
             (Default value = 0)

        Returns
        -------

        """
        k = k * units.Mpc
        P = self.__nonlin_power(k, z)
        return P * units.Mpc**3

    def __nonlin_electron_power(self, k, z=0):
        be = self.__free_electrons_bias(k, z)
        P = self.__nonlin_power(k, z)
        return be * P

    def nonlin_electron_power(self, k, z=0):
        """

        Parameters
        ----------
        k :

        z :
             (Default value = 0)

        Returns
        -------

        """
        ki = k * units.Mpc
        P = self.__nonlin_electron_power(ki, z)
        return P * units.Mpc**3

    def __dm_igm_integral(self, z, kmin=0.0, kmax=numpy.inf):

        def integrand(k):
            """

            Parameters
            ----------
            k :


            Returns
            -------

            """
            return k * self.__nonlin_electron_power(k, z)
        integral, _ = quad(integrand, kmin, kmax, limit=100, epsrel=1.49e-7)
        return integral

    def dm_igm_integral(self, z, kmin=0.0, kmax=numpy.inf, unit=units.Mpc):
        """

        Parameters
        ----------
        z :

        kmin :
             (Default value = 0.0)
        kmax :
             (Default value = numpy.inf)
        unit :
             (Default value = units.Mpc)

        Returns
        -------

        """
        func = numpy.vectorize(self.__dm_igm_integral,
                               excluded=['kmin', 'kmax'])
        return func(z, kmin, kmax) * unit / (2 * numpy.pi)

    def Hubble(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        E = self.h_over_h0(z)
        return self.H0 * E

    def differential_comoving_volume(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        r = self.comoving_radial_distance(z)
        Dh = constants.c / self.Hubble(z)
        dcoVol = (Dh * r**2).to(units.Mpc**3)
        return dcoVol / units.sr

    def star_formation_rate(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

        """
        num = 0.017 + 0.13 * z
        dem = 1 + (z / 3.3)**5.3

        return (num / dem) * units.Msun / units.yr / units.Mpc**3

    @cached_property
    def critical_density0(self):
        """ """
        rho_c0 = 3 * self.H0**2 / (8 * numpy.pi * constants.G)
        return rho_c0.to(units.g / units.cm**3)

    @cached_property
    def hubble_distance(self):
        """ """
        Dh = constants.c / self.H0
        return Dh.to(units.Mpc)

    @cached_property
    def baryon_number_density(self):
        """ """
        Omega_b = self.Omega_b
        rho_c0 = self.critical_density0
        n = Omega_b * rho_c0 / constants.m_p
        return n.to(1 / units.cm**3)
