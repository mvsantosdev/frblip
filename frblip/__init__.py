import warnings

from .grid import CartesianGrid
from .pattern import FunctionalPattern
from .fast_radio_bursts import FastRadioBursts
from .radio_telescope import RadioTelescope
from .observation import density_flux, interferometry_density_flux


warning_message = ''.join([
    '\n\nFRBlip is a beta version yet, and',
    'is available only for BINGO members.',
    '\nPlease do not use it for any other',
    'goals or share the code with someone outside the colaboration.',
    '\nReport any bug to mvsantos_at_protonmail.com\n'
])

warnings.warn(warning_message)
