# spotpy
A Statistical Parameter Optimization Framework for Python

---

[![PyPI Downloads][pypi-dl-image]][pypi-dl-link]
[![PyPI Version][pypi-v-image]][pypi-v-link]
[![Python Versions][pypi-pyv-image]][pypi-pyv-link]
[![Build Status][travis-image]][travis-link]
[![License][license-image]][license-link]

[pypi-dl-image]: https://img.shields.io/pypi/dm/spotpy.png
[pypi-dl-link]: https://pypi.python.org/pypi/spotpy
[pypi-v-image]: https://img.shields.io/pypi/v/spotpy.png
[pypi-v-link]: https://pypi.python.org/pypi/spotpy
[pypi-pyv-image]: https://img.shields.io/pypi/pyversions/spotpy.png
[pypi-pyv-link]: https://img.shields.io/pypi/pyversions/spotpy
[travis-image]: https://img.shields.io/travis/thouska/spotpy/master.png
[travis-link]: https://travis-ci.org/thouska/spotpy
[license-image]: https://img.shields.io/badge/license-MIT-blue.png
[license-link]: http://opensource.org/licenses/MIT


=================
Purpose
=================

SPOTPY is a Python tool that enables the use of Computational optimization techniques for calibration, uncertainty 
and sensitivity analysis techniques of almost every (environmental-) model.
 
The simplicity and flexibility enables the use and test of different 
algorithms without the need of complex codes::

	sampler = spotpy.algorithms.sceua(model_setup())     # Initialize your model with a setup file
	sampler.sample(10000)                                # Run the model
	results = sampler.getdata()                          # Load the results
	spotpy.analyser.plot_parametertrace(results)         # Show the results


=================
Features
=================

Complex algorithms bring complex tasks to link them with a model. 
We want to make this task as easy as possible. 
Some features you can use with the SPOTPY package are:

* Fitting models to evaluation data with different algorithms. 
  Available algorithms are: 
  
  * Monte Carlo (`MC`)
  * Markov-Chain Monte-Carlo (`MCMC`)
  * Maximum Likelihood Estimation (`MLE`)
  * Latin-Hypercube Sampling (`LHS`) 
  * Simulated Annealing (`SA`)
  * Shuffled Complex Evolution Algorithm (`SCE-UA`)
  * Differential Evolution Adaptive Metropolis Algorithm (`DE-MCz`) 
  * RObust Parameter Estimation (`ROPE`).
  * Fourier Amplitude Sensitivity Test (`FAST`)

* Wide range of objective functions (also known as loss function, fitness function or energy function) to validate the sampled results. Available functions are

  * Bias
  * Nash-Sutcliff (`NSE`)
  * logarithmic Nash-Sutcliff (`logNSE`)
  * logarithmic probability (`logp`)
  * Correlation Coefficient (`r`)
  * Coefficient of Determination (`r^2`)
  * Mean Squared Error (`MSE`)
  * Root Mean Squared Error (`RMSE`)
  * Mean Absolute Error (`MAE`)
  * Relative Root Mean Squared Error (`RRMSE`)
  * Agreement Index (`AI`)
  * Covariance, Decomposed MSE (`dMSE`).

* Prebuild parameter distribution functions: 

  * Uniform
  * Normal
  * logNormal
  * Chisquare
  * Exponential
  * Gamma
  * Wald
  * Weilbull

* Wide range to adapt algorithms to perform uncertainty-, sensitivity analysis or calibration
  of a model.

* Multi-objective support
 
* MPI support for fast parallel computing

* A progress bar monitoring the sampling loops. Enables you to plan your coffee brakes.

* Use of NumPy functions as often as possible. This makes your coffee brakes short.

* Different databases solutions: `ram` storage for fast sampling a simple , `csv` tables
  the save solution for long duration samplings.

* Automatic best run selecting and plotting

* Parameter trace plotting

* Parameter interaction plot including the Gaussian-kde function

* Regression analysis between simulation and evaluation data

* Posterior distribution plot

* Convergence diagnostics with Gelman-Rubin and the Geweke plot

=================
Install
=================

Installing SPOTPY is easy. Just use:

	pip install spotpy

Or, after downloading the [source code](https://pypi.python.org/pypi/spotpy "source code") and making sure python is in your OS path:

	python setup.py install
	
=================
Support
=================

* Documentation: http://www.uni-giessen.de/cms/faculties/f09/institutes/ilr/hydro/download/spotpy

* Feel free to contact the authors of this tool for any support questions.

* Please contact the authors in case of any bug.

* If you use this package for a scientific research paper, please cite SPOTPY. It is peer-reviewed.

* Patches/enhancements and any other contributions to this package are very welcome!

* GitHub: https://github.com/thouska/spotpy

=================
Version 1.1.0
=================

* Changed `likelihood` to `objectivefunction`. Checkout new [example spotpy_setup files](http://fb09-pasig.umwelt.uni-giessen.de/spotpy/Tutorial/2-Rosenbrock/#creating-the-setup-file).  

=================
Version 1.1.1 - 1.1.3
=================

* Minor bugfixes

=================
Version 1.1.4
=================

* Added a new functionality to sample parameters from a given list. Checkout the corresponding [example tutorial](http://fb09-pasig.umwelt.uni-giessen.de/spotpy/Tutorial/6-Advanced_hints/#sampling-from-a-given-parameter-list) and [code](https://github.com/thouska/spotpy/blob/master/spotpy/examples/tutorial_Parameterlist_iterator.py)

=================
Version 1.2.0
=================

* Added Python 3 support

=================
Version 1.2.1
=================

* Spotpy supports now userdefined databases. Checkout the corresponding [example tutorial](http://fb09-pasig.umwelt.uni-giessen.de/spotpy/Tutorial/6-Advanced_hints/#create-a-own-database) and [code](https://github.com/thouska/spotpy/blob/master/spotpy/examples/tutorial_ownDatabase.py).
