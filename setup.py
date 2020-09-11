from setuptools import setup

bla = setup(
    name='frblip',
    version='0.0.1',
    description='Fast Radio Burst mock catalogs synthesis.',
    author='Marcelo Vargas dos Santos',
    author_email='mvsantos_at_protonmail.com',
    packages=['frblip'],
    include_package_data=True,
    package_data={
        'frblip': [
            'data/*.npy',
            'data/*.npz',
            'data/*.csv'
        ]
    },
    python_requires='>=3.4',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'astropy',
        'healpy'
    ]
)
