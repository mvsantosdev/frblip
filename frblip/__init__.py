import warnings

from .fast_radio_bursts import blips, load
from .fast_radio_bursts import FastRadioBursts
from .healpix import HealPixMap
from .radio_telescope import RadioTelescope


__version__ = '0.0.1'


warning_message = ''.join([
    '\n\nFRBlip is a beta version yet, and',
    'is available only for BINGO members.',
    '\nPlease do not use it for any other',
    'goals or share the code with someone outside the colaboration.',
    '\nReport any bug to mvsantos_at_protonmail.com\n'
])

warnings.warn(warning_message)
