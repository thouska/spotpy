from setuptools import setup

setup(
  name = 'spotpy',
  version = '1.2.11',
  description = 'A Statistical Parameter Optimization Tool',

  author = 'Tobias Houska, Philipp Kraft, Alejandro Chamorro-Chavez and Lutz Breuer',
  author_email = 'tobias.houska@umwelt.uni-giessen.de',
  url = 'http://www.uni-giessen.de/cms/faculties/f09/institutes/ilr/hydro/download/spotpy',
  #download_url = 'svn://fb09-pasig.umwelt.uni-giessen.de/spotpy/trunk/', 
  license = 'MIT',
  packages = ["spotpy", "spotpy.examples", "spotpy.algorithms", "spotpy.parallel"],
  include_package_data = True,
  #py_modules = ["spotpy"], #"spotpy.examples", "spotpy.algorithms", "spotpy.parallel"],
  #test_suite = 'spotpy.examples',
  use_2to3 = True,
  #**extra,
  #long_description="""
  #          This package enables to use of powerful uncertainty, calibration and sensitivity analysis techniques in Python.
  #          If you are using SPOTPY please cite the following paper:
#
#            Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015.
#
#            """,
  keywords = ['Monte Carlo', 'MCMC','MLE', 'SCE-UA', 'Simulated Annealing', 'DE-MCz', 'ROPE', 'Uncertainty', 'Calibration', 'Model'],
  classifiers = [
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
        )