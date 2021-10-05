import numpy

from scipy.stats import truncnorm
from scipy.integrate import odeint
from astropy import units, constants

unit = units.pc / units.cm**3


class IGM():

    def __init__(self, cosmology, zmin=0.0, zmax=6.2, Xp=0.76,
                 a=0.475, b=0.703, c=3.19, z0=5.42):

        Yp = 1 - Xp

        self.a = a * (Xp + Yp / 2)
        self.b = b
        self.c = c
        self.z0 = z0

        self.zmin = zmin
        self.zmax = min(zmax, 6.43)

        self.cosmology = cosmology

        self.__integrate()

    def __fe(self, z):
        c1 = self.a * (z + self.b)**0.02
        c2 = 1 - numpy.tanh(self.c * (z - self.z0))
        return c1 * c2

    def __W(self, z):
        fe = self.__fe(z)
        return (1 + z) * fe

    def __integrand(self, X, z):

        X1, X2 = X
        h_over_h0 = self.cosmology.h_over_h0(z)
        dm_int = self.cosmology.dm_igm_integral(z, unit=1)

        W = self.__W(z)

        dX1 = W / h_over_h0
        dX2 = dX1 * W * dm_int

        return dX1, dX2

    def __integrate(self):

        Dh = self.cosmology.hubble_distance
        ne = self.cosmology.baryon_number_density

        self.redshift = numpy.linspace(self.zmin, self.zmax, 100)
        X = odeint(self.__integrand, (0, 0), self.redshift)
        self.mean_dm = Dh * ne * X[:, 0]
        self.std_dm = ne * numpy.sqrt(Dh * X[:, 1] * units.Mpc)

        self.mean_dm = self.mean_dm.to(unit)
        self.std_dm = self.std_dm.to(unit)

    def mean(self, z):
        return numpy.interp(x=z, xp=self.redshift, fp=self.mean_dm)

    def std(self, z):
        return numpy.interp(x=z, xp=self.redshift, fp=self.std_dm)

    def __call__(self, z):
        loc = self.mean(z)
        scale = self.std(z)
        z = truncnorm.rvs(a=-loc/scale, b=numpy.inf, size=z.size)
        return scale * z + loc
