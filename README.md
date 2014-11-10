EPwx - sparse PCA using EP
==========================

Introduction
------------

Matlab code for Bayesian sparse principal component analysis with Gaussian and/or probit likelihoods and spike and slab sparse prior. Inference methods:

 * Expectation propagation (EP),
 * Hybrid variational bayes - EP (VB-EP),
 * Gibbs sampling.

Warning: The code is not "production quality".

Installation
------------

EP and Gibbs sampling use C++ code, which needs to be compiled in Matlab using mex. The code requires [Eigen matrix library][Eigen]. Eigen doesn't require any installation: just download and unzip it. To compile the C++ code, type in Matlab command line (after replacing "/path/to/eigen/" with the location of the unzipped Eigen library):

```
mex -largeArrayDims -I/path/to/eigen/ ep_wx_parallelep_factcov.cpp
mex CXXFLAGS="\$CXXFLAGS -std=c++0x" -largeArrayDims -I/path/to/eigen/ gibbs_wx_probit_half_normal_sampling_nomatlab.cpp
```

  [Eigen]: http://eigen.tuxfamily.org

Usage
-----

See example.m for an example.

Reference
---------

Peltola, Jylänki, Vehtari. _Expectation propagation for likelihoods depending on an inner product of two multivariate random variables._ In JMLR Workshop and Conference Proceedings: AISTATS 2014, volume 33, p. 769-777. ([link][])

The EP-VB hybrid Gibbs sampling are described in Rattray, Stegle, Sharp, Winn (2009) Inference algorithms and learning theory for Bayesian sparse factor analysis. Journal of Physics: Conference Series, 197(1).

  [link]: http://jmlr.org/proceedings/papers/v33/peltola14.html

Acknowledgements
----------------

Code for the truncated normal sampling used in the Gibbs sampling for probit likelihood is by Nicolas Chopin. The original code is available at https://sites.google.com/site/nicolaschopinstatistician/software and has been adapted for the mex-file.

Changes
-------

2014-04-16 (first release)
 * The code has been slightly updated from the version used in the AISTATS2014 article, but should give similar results.
 * Minor updates to EP and VB-EP approximation initializations.
 * EP uses Newton iterations to refine the end-point for the numerical integration.

Contact
-------

Tomi Peltola, tomi.peltola@aalto.fi
http://becs.aalto.fi/en/research/bayes/epwx/

License
-------

GPLv2, see http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt.
