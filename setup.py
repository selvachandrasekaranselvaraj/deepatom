from setuptools import setup, find_packages

setup(
    name='deepatom',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pymatgen',
        'gemd',
        'ase',
        'mbtr',
        'torch',
        'torch-geometric',
        'lammps',
    ],
    entry_points={
        'console_scripts': [
            'deepatom=deepatom.main:main',
        ],
    },
    test_suite='tests',
    python_requires='>=3.6',
)

