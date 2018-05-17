# Copyright (c) 2015, Tobias Houska

from setuptools import setup
import os

setup(
  name = 'spotpy',
  version = '1.3.27',
  description = 'A Statistical Parameter Optimization Tool',
  long_description=open(os.path.join(os.path.dirname(__file__),
                                       "README.rst")).read(),
  author = 'Tobias Houska, Philipp Kraft, Alejandro Chamorro-Chavez and Lutz Breuer',
  author_email = 'tobias.houska@umwelt.uni-giessen.de',
  url = 'http://www.uni-giessen.de/cms/faculties/f09/institutes/ilr/hydro/download/spotpy',
  license = 'MIT',
  packages = ["spotpy", "spotpy.examples", "spotpy.examples.hymod_python", "spotpy.examples.hymod_exe", "spotpy.algorithms", "spotpy.parallel", "spotpy.gui", "spotpy.hydrology"],
  package_data={
   'spotpy.examples.hymod_exe': ['*'],
   'spotpy.examples.hymod_python': ['*'],
   },
  #include_package_data = True,
  use_2to3 = True,
  keywords = 'Monte Carlo, MCMC, MLE, SCE-UA, Simulated Annealing, DE-MCz, DREAM, ROPE, Artifical Bee Colony, Uncertainty, Calibration, Model, Signatures',
  classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules'],
        )
