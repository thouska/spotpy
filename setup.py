from setuptools import setup
import sys

extra = {}
if sys.version_info >= (3,):
    extra['use_2to3'] = True

  setup(
  name = 'spotpy',
  version = '1.2.0',
  description = 'A Statistical Parameter Optimization Tool',

  author = 'Tobias Houska, Philipp Kraft, Alejandro Chamorro-Chavez and Lutz Breuer',
  author_email = 'tobias.houska@umwelt.uni-giessen.de',
  url = 'http://www.uni-giessen.de/cms/faculties/f09/institutes/ilr/hydro/download/spotpy',
  download_url = 'svn://fb09-pasig.umwelt.uni-giessen.de/spotpy/trunk/', 
  license='MIT',
  packages=["spotpy", "spotpy.examples", "spotpy.algorithms", "spotpy.parallel"],
  include_package_data=True,
  keywords = ['Monte Carlo', 'MCMC','MLE', 'SCE-UA', 'Simulated Annealing', 'DE-MCz', 'ROPE', 'Uncertainty', 'Calibration', 'Model'],
  classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Software Development :: Libraries :: Python Modules'],
        **extra,
        )