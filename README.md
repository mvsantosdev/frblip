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

## Warning: Alpha Version Software

Dear Users,

Please be advised that the software you are about to use is currently in the alpha development stage. This means that it is a preliminary release and may not be stable or feature-complete. As such, it is intended for testing purposes only, and we strongly discourage its use in a production or critical environment.

Key points to consider:

1.  Bugs and Issues: Alpha versions are likely to contain bugs, errors, and unforeseen issues. These can range from minor inconveniences to serious malfunctions.

2.  Limited Features: Some features may be missing or incomplete in the alpha version. Expect changes, additions, and removals in subsequent updates.

3.  Data Loss Risk: Due to the experimental nature of alpha software, there is a risk of data loss or corruption. Avoid using it with important or irreplaceable data.

4.  Security Concerns: Security measures may not be fully implemented in alpha versions. Do not use this software for tasks that require a high level of security.

5.  Frequent Updates: Alpha software is actively being developed, and updates may be frequent. Be prepared for regular updates and changes to improve the software.

6.  User Feedback: Your feedback is crucial in helping us identify and address issues. Please report any bugs, glitches, or suggestions for improvement through the provided channels.

By using this alpha version, you acknowledge and accept the risks associated with its experimental nature. We appreciate your participation in the testing process and look forward to your valuable feedback to enhance the software.

Thank you for your understanding.
