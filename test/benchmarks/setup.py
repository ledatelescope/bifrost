#!/usr/bin/env python
"""Installation script for some shared benchmarking utilities
"""

from setuptools import setup

setup(name='bifrost_benchmarks',
      version='0.0',
      description='Shared benchmarking utility',
      url='github.com/ledatelescope/bifrost',
      packages=['bifrost_benchmarks'],
      license='BSD-3-Clause',
      package_dir={'bifrost_benchmarks':'bifrost_benchmarks'},
      install_requires=[
          'bifrost',
          'lizard'
      ])
