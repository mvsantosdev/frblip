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
    python_requires='>=3.8',
    install_requires=[
        'dill',
        'numpy',
        'pandas',
        'xarray',
        'scipy',
        'astropy',
        'astropy-healpix',
        'healpy',
        'pygedm',
        'camb',
        'pyccl'
    ]
)
