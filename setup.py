# Copyright (c) 2015, Tobias Houska

from setuptools import setup, find_packages
import os

setup(
  name = 'spotpy',
  version = '1.5.13',
  description = 'A Statistical Parameter Optimization Tool',
  long_description=open(os.path.join(os.path.dirname(__file__),
                                       "README.rst")).read(),
  author = 'Tobias Houska, Philipp Kraft, Alejandro Chamorro-Chavez and Lutz Breuer',
  author_email = 'tobias.houska@umwelt.uni-giessen.de',
  url = 'https://spotpy.readthedocs.io/en/latest/',
  license = 'MIT',
  install_requires=[
          'scipy', 'numpy'],
  packages=find_packages(exclude=["tests*", "docs*"]),
  use_2to3 = True,
  keywords = 'Monte Carlo, MCMC, MLE, SCE-UA, Simulated Annealing, DE-MCz, DREAM, ROPE, Artifical Bee Colony, DDS, PA-DDS, Uncertainty, Calibration, Model, Signatures',
  classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries :: Python Modules'],
        )
