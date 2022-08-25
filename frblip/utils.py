import bz2
import dill

from .fast_radio_bursts import FastRadioBursts
from .radio_telescope import RadioTelescope


blips = FastRadioBursts
telescope = RadioTelescope


def load(file):
    """

    Parameters
    ----------
    file :


    Returns
    -------

    """
    return FastRadioBursts.load(file)


def load_catalog(name):
    """

    Parameters
    ----------
    name :


    Returns
    -------

    """

    file = bz2.BZ2File(name, 'rb')
    catalog = dill.load(file)
    file.close()
    return catalog
