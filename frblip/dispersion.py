import os

import numpy
import healpy

import pandas

from astropy import units, constants, cosmology

from scipy.integrate import cumtrapz

from .utils import _DATA


unit = units.parcsec / units.cm**3

mp = constants.m_p
cspeed = constants.c


def galactic_dispersion(lon, lat, nside=256):

    ne2001map = numpy.load('{}/ne2001map{}.npy'.format(_DATA, nside))

    return unit * healpy.get_interp_val(ne2001map, lon.value,
                                        lat.value, lonlat=True)


def host_galaxy_dispersion(z, logrange=(-1, 4), model='ALGs(YMW16)'):

    pars = pandas.read_csv('{}/HostGalaxyDM.csv'.format(_DATA),
                           index_col='Parameters').loc[:, model]

    a1, a2 = pars.a1, pars.a2
    b1, b2 = pars.b1, pars.b2
    c1, c2 = pars.c1, pars.c2

    C = (a1 * c1 + a2 * c2) * numpy.sqrt(numpy.pi)

    logDMs = numpy.linspace(*logrange, 100)

    U = numpy.random.random(len(z))

    pdf1 = a1 * numpy.exp(- ((logDMs - b1)/c1)**2)
    pdf2 = a2 * numpy.exp(- ((logDMs - b2)/c2)**2)

    pdf = (pdf1 + pdf2) / C

    cdflogDM = cumtrapz(x=logDMs, y=pdf, initial=0)

    q = 1 + z

    SFR = q**2.7 / (1 + 0.00257 * q**5.6)

    logDM = numpy.interp(x=U, xp=cdflogDM, fp=logDMs)

    return numpy.sqrt(SFR) * (10**logDM) * unit


def source_dispersion(size, DM_min=0, DM_max=50):

    return numpy.random.uniform(low=DM_min, high=DM_max, size=size) * unit


def igm_dispersion(z, zmax, zmin=0, fIGM=0.83,
                   cosmo=cosmology.Planck18_arXiv_v2):

    H0 = cosmo.H0
    Omega_b0 = cosmo.Ob0
    rho_crit0 = cosmo.critical_density0

    zs = numpy.linspace(zmin, zmax, 100)

    gh = 0.75
    ghe = numpy.full_like(zs, 0.125)

    idx = zs > 3.0
    zhe = zs[idx]

    ghe[idx] *= 0.025 * zhe**3 - 0.244 * zhe**2 + 0.513 * zhe + 1.006

    g = gh + ghe

    C_igm = (cspeed * rho_crit0 * Omega_b0 * fIGM / H0 / mp)
    C_igm = C_igm.to(units.pc / units.cm**3).value

    dI = g * (1 + zs) * cosmo.inv_efunc(zs)

    Iigm = cumtrapz(x=zs, y=dI, initial=0)

    return C_igm * numpy.interp(x=z, xp=zs, fp=Iigm) * unit
