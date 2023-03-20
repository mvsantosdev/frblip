from functools import cached_property

import numpy
from astropy import coordinates, units
from scipy.special import j1

from .decorators import default_units


class FunctionalPattern(object):
    @default_units(radius='deg', alt='deg', az='deg')
    def __init__(
        self,
        radius: units.Quantity,
        alt: float = 90.0,
        az: float = 0.0,
        kind: str = 'gaussian',
    ):

        self.radius = radius
        self.alt = alt
        self.az = az

        self.beams = numpy.unique([self.alt.size, self.az.size])
        assert self.beams.size == 1
        self.beams = self.beams.item()
        self.offs = self.beams

        if (self.radius.size > 1) and (self.beams == 1):
            self.beams = self.radius.size

        if (self.beams > 1) and (self.radius.size == 1):
            self.radius = numpy.tile(self.radius, self.beams)

        self.response = getattr(self, f'_{kind}')

    @cached_property
    def offsets(self) -> coordinates.SkyOffsetFrame:

        altaz = coordinates.AltAz(alt=self.alt, az=self.az)
        return coordinates.SkyOffsetFrame(origin=altaz)

    def __call__(self, altaz) -> numpy.ndarray:

        altazoffs = [
            altaz.transform_to(self.offsets[i]) for i in range(self.offs)
        ]

        cossines = numpy.column_stack(
            [altazoff.cartesian.x for altazoff in altazoffs]
        )

        arcs = numpy.arccos(cossines)
        rescaled_arc = (arcs / self.radius).to(1).value
        return self.response(rescaled_arc)

    def _tophat(self, x: float | numpy.ndarray) -> numpy.ndarray:
        return (numpy.abs(x) <= 1).astype(int)

    def _gaussian(self, x: float | numpy.ndarray) -> numpy.ndarray:
        return numpy.exp(-(x**2))

    def _bessel(self, x: float | numpy.ndarray) -> numpy.ndarray:
        return numpy.nan_to_num(j1(2 * x) / x, nan=1.0) ** 2
