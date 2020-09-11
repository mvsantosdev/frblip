import numpy

from scipy.special import j1


patterns = {
    'tophat': lambda x: (numpy.abs(x) <= 1).astype(numpy.float),
    'gaussian': lambda x: numpy.exp(-x**2),
    'bessel': lambda x: (j1(2 * x) / x)**2
}
