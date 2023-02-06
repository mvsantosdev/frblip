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
>>> conda create -n env_name
>>> conda activate env_name
```

The easiest way to install `pyccl` and `pygedm`, dependencies, is by using conda (it may take some minutes):

```
>>> (env_name) conda install -c conda-forge pyccl pygedm
```

To install the remaining dependencies and frblip:

```
>>> (env_name) cd frblip
>>> (env_name) pip install -r requirements.txt
>>> (env_name) pip install -e .
```

## Examples notebooks

1. **[Quick Start](https://github.com/mvsantosdev/frblip/tree/master/examples/quick_start.ipynb)**

2. **[All examples](https://github.com/mvsantosdev/frblip/tree/master/examples)**