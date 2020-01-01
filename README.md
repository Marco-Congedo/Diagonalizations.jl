# Diagonalizations.jl

| **Documentation**  | 
|:---------------------------------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://Marco-Congedo.github.io/Diagonalizations.jl/dev) |

![](/docs/src/assets/FiggMCA.png)

**Diagonalizations.jl** is a [**Julia**](https://julialang.org/) signal processing package implementing the following *closed form* and *iterative* diagonalization procedures:

| Acronym   | Full Name | Datasets ( *m* ) | Observations ( *k* ) |
|:----------|:---------:|:---------:|:---------:|
| PCA | Principal Component Analysis | 1 | 1 |
| Whitening | Whitening (Sphering) | 1 | 1 |
| MCA | Maximum Covariance Analysis | 2 | 1 |
| CCA | Canonical Correlation Analysis | 2 | 1 |
| gMCA | generalized MCA | >1 | 1 |
| gCCA | generalized CCA | >1 | 1 |
| CSP | Common Spatial Pattern | 1 | 2 |
| CSTP | Common Spatio-Temporal Pattern | 1 | >1 |
| AJD | Approximate Joint Diagonalization | 1 | >1 |
| mAJD | multiple AJD | >1 | >1 |

As compared to [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl)
this package supports :
- the `dims` keyword
- shrinkage covariance matrix estimations throught package [CovarianceEstimation](https://github.com/mateuszbaran/CovarianceEstimation.jl)
- average covariance estimations using metrics for the manifold of positive definite matrices using the [PosDefManifold](https://github.com/Marco-Congedo/PosDefManifold.jl) package
- facilities to set the subspace dimension upon construction
- diagonalization procedures for the case ``m≥2`` and ``k≥2``.

## Installation

To install the package execute the following command in Julia's REPL:

    ]add CovarianceEstimation PosDefManifold Diagonalizations

## Disclaimer

This package is throughoutly tested for real data input. Testing for complex data input is still to be done.

## Examples

```
using Diagonalizations, Test
n, t=10, 100

# generate an nxt data matrix
X=genDataMatrix(n, t)

pX=pca(X)

# compute the covariance matrix
C=Symmetric((X*X')/t)

# the following is an equivalent constructor
pC=pca(C)

@test C≈pC.F*pC.D*pC.F'  
@test pX==pC # the output of two constructors is equivalent
```

## About the Authors

[Marco Congedo](https://sites.google.com/site/marcocongedo), corresponding
author, is a research scientist of [CNRS](http://www.cnrs.fr/en) (Centre National de la Recherche Scientifique), working in [UGA](https://www.univ-grenoble-alpes.fr/english/) (University of Grenoble Alpes). **contact**: marco *dot* congedo *at* gmail *dot* com

| **Documentation**  | 
|:---------------------------------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://Marco-Congedo.github.io/Diagonalizations.jl/dev) |
