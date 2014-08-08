import sys

if sys.hexversion < 0x2060000:
    raise NotImplementedError('Python < 2.6 not supported.')

import ez_setup
ez_setup.use_setuptools()

from setuptools import setup

with open('README.rst') as file:
    long_description = file.read()

if sys.hexversion < 0x2070000:
    requires = ['numpy', 'h5py (>= 2.1)', 'hdf5storage', 'ordereddict']
else:
    requires = ['numpy', 'h5py (>= 2.1)', 'hdf5storage']

setup(name='pydons',
      version='0.1.0',
      description='Python data manimulation add-ons',
      long_description=long_description,
      author='Jakub Urban',
      author_email='coobas at gmail dt com',
      url='https://github.com/coobas/pydons',
      packages=['pydons'],
      requires=requires,
      license='MIT',
      keywords='hdf5 matlab',
      classifiers=[
          "Programming Language :: Python :: 2.6",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Development Status :: 3 - Alpha",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Intended Audience :: Developers",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering",
          "Topic :: Database",
          "Topic :: Software Development :: Libraries :: Python Modules"
      ],
      test_suite='nose.collector',
      tests_require='nose>=1.0'
      )
