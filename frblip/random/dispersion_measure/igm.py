import numpy

from scipy.stats import truncnorm
from scipy.integrate import odeint
from astropy import units

from ...cosmology import Cosmology

unit = units.pc / units.cm**3


class InterGalacticDM():

    MODELS = {
        'Takahashi2021': {
            'a': 0.475, 'b': 0.703, 'c': 3.19, 'd': 0.02, 'z0': 5.42
        }
    }

    def __init__(
        self,
        free_electron_model: str = 'Takahashi2021',
        cosmology: str = 'Planck_18',
        zmin: float = 0.0,
        zmax: float = 6.2,
        Xp: float = 0.76,
        **kwargs
    ):

        self.Xp = Xp
        self.Yp = 1 - Xp
        self.figm = (1 + Xp) / 2

        for key, value in self.MODELS[free_electron_model].items():
            self.__dict__[key] = kwargs.get(key, value)
        self._fe = getattr(self, f'_{free_electron_model.lower()}')

        self.zmin = zmin
        self.zmax = min(zmax, 6.43)

        if isinstance(cosmology, str):
            kw = {
                'name': cosmology,
                'free_electron_bias': free_electron_model
            }
            self.cosmology = Cosmology(**kw)
        elif isinstance(cosmology, Cosmology):
            self.cosmology = cosmology

        self._integrate()

    def _takahashi2021(self, z: numpy.ndarray) -> numpy.ndarray:
        c1 = self.a * (z + self.b)**self.d
        c2 = 1 - numpy.tanh(self.c * (z - self.z0))
        return c1 * c2

    def _W(self, z: numpy.ndarray) -> numpy.ndarray:
        fe = self.figm * self._fe(z)
        return (1 + z) * fe

    def _integrand(
        self,
        X: tuple[float, float],
        z: float
    ) -> tuple[float, float]:

        X1, X2 = X
        h_over_h0 = self.cosmology.h_over_h0(z)
        dm_int = self.cosmology.dm_igm_integral(z, unit=1)

        W = self._W(z)

        dX1 = W / h_over_h0
        dX2 = dX1 * W * dm_int

        return dX1, dX2

    def _integrate(self):

        Dh = self.cosmology.hubble_distance
        ne = self.cosmology.baryon_number_density

        self.redshift = numpy.linspace(self.zmin, self.zmax, 100)
        X = odeint(self._integrand, (0, 0), self.redshift)
        self.mean_dm = Dh * ne * X[:, 0]
        self.std_dm = ne * numpy.sqrt(Dh * X[:, 1] * units.Mpc)

        self.mean_dm = self.mean_dm.to(unit)
        self.std_dm = self.std_dm.to(unit)

    def mean(self, z: numpy.ndarray) -> numpy.ndarray:

        return numpy.interp(x=z, xp=self.redshift, fp=self.mean_dm)

    def std(self, z: numpy.ndarray) -> numpy.ndarray:

        return numpy.interp(x=z, xp=self.redshift, fp=self.std_dm)

    def __call__(self, z: numpy.ndarray) -> numpy.ndarray:
        loc = self.mean(z)
        scale = self.std(z)
        z = truncnorm.rvs(a=-loc/scale, b=numpy.inf, size=z.size)
        return scale * z + loc
