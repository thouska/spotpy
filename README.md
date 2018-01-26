# spotpy
A Statistical Parameter Optimization Tool for Python

---

[![PyPI Version][pypi-v-image]][pypi-v-link]
[![Python Versions][pypi-pyv-image]][pypi-pyv-link]
[![Build Status][travis-image]][travis-link]
[![License][license-image]][license-link]
[![Coverage Status][coverage-image]][coverage-link]

[pypi-v-image]: https://img.shields.io/pypi/v/spotpy.png
[pypi-v-link]: https://pypi.python.org/pypi/spotpy
[pypi-pyv-image]: https://img.shields.io/pypi/pyversions/spotpy.png
[pypi-pyv-link]: https://img.shields.io/pypi/pyversions/spotpy
[travis-image]: https://img.shields.io/travis/thouska/spotpy/master.png
[travis-link]: https://travis-ci.org/thouska/spotpy
[license-image]: https://img.shields.io/badge/license-MIT-blue.png
[license-link]: http://opensource.org/licenses/MIT
[coverage-image]: https://coveralls.io/repos/github/thouska/spotpy/badge.svg?branch=master
[coverage-link]: https://coveralls.io/github/thouska/spotpy?branch=master



Purpose
=================

SPOTPY is a Python framework that enables the use of Computational optimization techniques for calibration, uncertainty 
and sensitivity analysis techniques of almost every (environmental-) model. The package is puplished in the open source journal PLoS One:

Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made Python Package, PLoS ONE, 
10(12), e0145180, doi:[10.1371/journal.pone.0145180](http://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0145180 "SPOTting Model Parameters Using a Ready-Made Python Package"), 2015
 
The simplicity and flexibility enables the use and test of different 
algorithms of almost any model, without the need of complex codes::

	sampler = spotpy.algorithms.sceua(model_setup())     # Initialize your model with a setup file
	sampler.sample(10000)                                # Run the model
	results = sampler.getdata()                          # Load the results
	spotpy.analyser.plot_parametertrace(results)         # Show the results



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
  * Differential Evolution Markov Chain Algorithm (`DE-MCz`)
  * Differential Evolution Adaptive Metropolis Algorithm (`DREAM`)
  * RObust Parameter Estimation (`ROPE`)
  * Fourier Amplitude Sensitivity Test (`FAST`)
  * Artificial Bee Colony (`ABC`)
  * Fitness Scaled Chaotic Artificial Bee Colony (`FSCABC`)

* Wide range of objective functions (also known as loss function, fitness function or energy function) to validate the sampled results. Available functions are

  * Bias
  * PBias
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
  * Covariance, Decomposed MSE (`dMSE`)
  * Kling-Gupta Efficiency (`KGE`).

* Prebuild parameter distribution functions: 

  * Uniform
  * Normal
  * logNormal
  * Chisquare
  * Exponential
  * Gamma
  * Wald
  * Weilbull

* A toolbox of hydrological signatures, to see a hydrological fit of modeled data to the measured one. 

To use this signatures, you call the _get_**HydrologicalSignature** function and entering the measured and the modeled data.
Setting `mode='calc_Dev'` performs our desired calculation. Sometimes additional parameters are needed, as for example
a timestamp list (in a pandas object) or threshold values. The closer the return value is to zero, the more fits the simulated 
data with measured one. 


	spotpy.signatures.getSlopeFDC(measured_data,results,mode='calc_Dev')

By changing the value of `mode` to `'get_signature'`, we get the signature value _only_ for the parameter `data`.

By changing the value of `mode` to `'get_raw_data'`, we get a raw calculation of flood-signatures, belonging to each signature calculation.

**tutorial_signatures.py** contains al available functions and it's mode settings.

  * List of available hydrological signatures, additional required parameters are in brackets
    * getMeanFlow
    * getMedianFlow
    * getSkewness
    * getCoeffVariation
    * getQ001
    * getQ01
    * getQ1
    * getQ5
    * getQ10
    * getQ20
    * getQ85
    * getQ95
    * getQ99
    * getSlopeFDC
    * getAverageFloodOverflowPerSection(datetime_series, threshold_value)
    * getAverageFloodFrequencyPerSection(datetime_series, threshold_value)
    * getAverageFloodDuration(datetime_series, threshold_value)
    * getAverageBaseflowUnderflowPerSection(datetime_series, threshold_value)
    * getAverageBaseflowFrequencyPerSection(datetime_series, threshold_value)
    * getFloodFrequency(datetime_series, threshold_value)
    * getBaseflowFrequency (datetime_series, threshold_value)
    * getLowFlowVar(datetime_series)
    * getHighFlowVar(datetime_series)
    * getBaseflowIndex(datetime_series)

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


Install
=================

Installing SPOTPY is easy. Just use:

	pip install spotpy

Or, after downloading the [source code](https://pypi.python.org/pypi/spotpy "source code") and making sure python is in your OS path:

	python setup.py install

	
Support
=================

* Documentation: http://www.uni-giessen.de/cms/faculties/f09/institutes/ilr/hydro/download/spotpy

* Feel free to contact the authors of this tool for any support questions.

* Please contact the authors in case of any bug.

* If you use this package for a scientific research paper, please cite SPOTPY. It is [peer-reviewed](http://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0145180 "SPOTting Model Parameters Using a Ready-Made Python Package").

* Patches/enhancements and any other contributions to this package are very welcome!


Getting started
=================
Have a look at https://github.com/thouska/spotpy/tree/master/spotpy/examples and http://fb09-pasig.umwelt.uni-giessen.de/spotpy/Tutorial/2-Rosenbrock/
