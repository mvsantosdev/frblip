import numpy

from sparse import COO
from astropy import units


_add_error = "Cannot add two quantities with different units."
_sub_error = "Cannot subtract two quantities with different units."


def get_unit(value):
    return getattr(value, 'unit', units.Unit(1))


def get_value(value):
    return getattr(value, 'value', value)


class SparseQuantity(COO):

    def __init__(self, data, coords=None, shape=None, unit=None):

        if coords is None:
            value = get_value(data)
            super().__init__(value)
        else:
            super().__init__(coords, data, shape)

        if unit is None:
            self.unit = get_unit(data)
        else:
            self.unit = unit

    @property
    def value(self):
        return COO(super().copy())
    
    @property
    def T(self):
        value = self.value.T
        return SparseQuantity(value, unit=self.unit)
    
    def abs(self):
        value = numpy.abs(self.value)
        return SparseQuantity(value, unit=self.unit)

    def apply(self, func):
        value = func(self.value)
        return SparseQuantity(value, unit=self.unit)

    def reshape(self, shape):
        value = self.value.reshape(shape)
        return SparseQuantity(value, unit=self.unit)

    def squeeze(self):
        shape = tuple(s for s in self.shape if s > 1)
        value = self.value.reshape(shape)
        return SparseQuantity(value, unit=self.unit)

    def todense(self):
        return self.value.todense() * self.unit

    def __add(self, other):
        assert self.unit == get_unit(other), _add_error
        value = self.value + get_value(other)
        return SparseQuantity(value, unit=self.unit)

    def __add__(self, other):
        return self.__add(other)

    def __radd__(self, other):
        return self.__add(other)

    def __sub__(self, other):
        assert self.unit == get_unit(other), _sub_error
        value = self.value - get_value(other)
        return SparseQuantity(value, unit=self.unit)

    def __mul(self, other):
        value = self.value * get_value(other)
        unit = self.unit * get_unit(other)
        return SparseQuantity(value, unit=unit)

    def __mul__(self, other):
        return self.__mul(other)

    def __rmul__(self, other):
        return self.__mul(other)

    def __truediv__(self, other):
        value = self.value / get_value(other)
        unit = self.unit / get_unit(other)
        return SparseQuantity(value, unit=unit)

    def __floordiv__(self, other):
        value = self.value // get_value(other)
        unit = self.unit / get_unit(other)
        return SparseQuantity(value, unit=unit)

    def __mod__(self, other):
        value = self.value % get_value(other)
        unit = self.unit / get_unit(other)
        return SparseQuantity(value, unit=unit)

    def __getitem__(self, idx):
        value = super().__getitem__(idx)
        return SparseQuantity(value, unit=self.unit)
