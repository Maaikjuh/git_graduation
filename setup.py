# -*- coding: utf-8 -*-
"""
A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
from setuptools import setup

from cvbtk import __version__

setup(
    # Name of the project, not the packages that it makes available.
    name='cvbtk',

    # Name of the package(s) that this project makes available.
    # This/these are what will be made importable.
    packages=['cvbtk'],

    # The version should be obtained from the package's __init__.py.
    version=__version__,

    # The short (one line) and long descriptions of the project.
    description='Cardiovascular Biomechanics Toolkit',
    long_description='Cardiovascular Biomechanics Toolkit',

    # Project homepage, or, canonical repository.
    url='https://bitbucket.org/ericnchen/leftventricle',

    # Primary developer(s) and contact information.
    author='Eric Chen',
    author_email='e.chen@tue.nl',

    # License this software is released under.
    license='Proprietary',

    # These packages are required to use all the features of this project.
    # FEniCS packages are not listed here as they must be manually installed.
    install_requires=['matplotlib', 'numpy', 'pandas', 'scipy'],

    # Additional dependencies depending on the installation type.
    # TODO Look into this and figure out how to take advantage of this.
    # extras_require={'dev': ['check-manifest'], 'test': ['coverage']},

    # Include the reference data files defined in MANIFEST.in.
    include_package_data=True
)
