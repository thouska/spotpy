# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased Changes](https://github.com/thouska/spotpy/compare/v1.6.3..master)


## Spotpy Version [v1.6.3](https://github.com/thouska/spotpy/compare/v1.6.0-rc1...v1.6.3) (2025-02-12)

* Introducing efast [#330](https://github.com/thouska/spotpy/pull/330)


## Spotpy Version [v1.6.0-rc1](https://github.com/thouska/spotpy/compare/v1.5.16..v1.6.0-rc1) (2022-07-15)

* move setuptools config to pyproject.toml following pep621
* Remove old Python2 necessarities [#287]
* Modernize package structure [#287]
* Spotpy source files moved to /src folder [#287]
* Automatized upload to Pypi and TestPypi [#287]
* New code style [Black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) and [isort](https://pypi.org/project/isort/)


## Spotpy Version [v1.5.16](https://github.com/thouska/spotpy/compare/v1.5.15..v.1.5.16) (2022-06-21)

* Update Tests to new Python3 standards
* Cleanup of old tests
* Introduce Github actions for testing and upload to pypi


## Spotpy Version [v1.5.15](https://github.com/thouska/spotpy/compare/v1.5.14..v.1.5.15) (2022-06-21)

* Using random number of chain pairs to generate the jump in Dream [#284]
* Minor improvements in NSGAii [#246]
* New Tutorial for [List sampling](https://github.com/thouska/spotpy/blob/master/spotpy/examples/tutorial_listsampler.py)
* Longer Strings allowed for parameter names (change from '|U30' to '|U100')
* Longer Floats by default in csv database (change from np.float16 to np.float32)
* Added random_state also for random package, which should better reproducability of spotpy results when set via sampler (not set as default) 


## Spotpy Version [v1.5.14](https://github.com/thouska/spotpy/compare/v1.5.13..v.1.5.14) (2020-10-09)

* New algorithm NSGAii [#246]
* Bugfix in generalizedLikelihoodFunction [#257]
* Bugfix in pickle file of sceua [#258]
* New documentation slides [#259]


## Spotpy Version [v1.5.13](https://github.com/thouska/spotpy/compare/v1.5.12...v1.5.13) (2020-09-07)

* Introducing package dependencies as requested [#249](https://github.com/thouska/spotpy/issues/249)
* Introducing chanchelog.md as requested [#225](https://github.com/thouska/spotpy/issues/225)
