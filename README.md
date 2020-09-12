# FRBlip

Code for FRB Mock simulations. Thisis a beta version only available only for BINGO members. Please do not use it for any other goals or share the code with someone outside the colaboration.
Report any bug to mvsantos_at_protonmail.com.

## Instalation

First clone the repository:

```
>>> git clone https://github.com/mvsantosdev/frblip.git
```

It is strongly recommended to create a exclusive enviorement, using conda for example

```
>>> conda create -n <env name>
>>> conda activate <env name>
>>> cd frblip
>>> pip install -r requirements.txt
>>> python setup.py install
```
        
## Quick Start

```
from frblip import CosmicBursts, RadioSurvey

# Mock catalog of random FRBs
frbs = CosmicBursts()

# to pandas DataFrame
df = frbs.to_pandas()

# Survey Object
bingo = RadioSurvey()

# Observations
coords, signal = bingo(frbs)

# coords: sky coordinates in survey site (astropy object)
# signal (n_beams, n_frbs, n_bands): telescope signal in Kelvin
```

