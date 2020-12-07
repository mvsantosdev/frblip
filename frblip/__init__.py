import warnings

from .cosmic_bursts import CosmicBursts
from .radio_telescope import RadioTelescope
from .observed_bursts import ObservedBursts, interferometry

from .utils import uv2azalt, azalt2uvw

warning_message = ''.join([
    '\n\nFRBlip is a beta version yet, and',
    'is available only for BINGO members.',
    '\nPlease do not use it for any other',
    'goals or share the code with someone outside the colaboration.',
    '\nReport any bug to mvsantos_at_protonmail.com\n'
])

warnings.warn(warning_message)
