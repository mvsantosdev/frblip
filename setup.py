from setuptools import setup, find_packages

bla = setup(
    name='frblip',
    version='0.0.1',
    description='Fast Radio Burst mock catalogs synthesis.',
    author='Marcelo Vargas dos Santos',
    author_email='mvsantos_at_protonmail.com',
    packages=find_packages(include=['frblip', 'frblip.*']),
    include_package_data=True,
    package_data={
        'frblip': [
            'data/*.npy',
            'data/*.npz',
            'data/*.csv'
        ]
    },
    python_requires='>=3.10',
    install_requires=[
        'toolz',
        'dill>=0.3.6',
        'numba>=0.55',
        'numpy>=1.21',
        'pandas',
        'sparse==0.13.0',
        'xarray==0.20.1',
        'scipy',
        'astropy',
        'astropy-healpix',
        'healpy',
        'pygedm',
        'camb',
        'pyccl>=2.3.0'
    ]
)
